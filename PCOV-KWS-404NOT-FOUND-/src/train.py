
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

from frontend import PCENFrontend
from model import PCOVKWS
from dataset import PCOVMultiTaskDataset
from loss import SVEmbeddingKDLoss

# ─────────────────────────────────────────────
# KAGGLE PATHS
# ─────────────────────────────────────────────
TRAIN_MANIFEST   = "/kaggle/working/pcov-kws-manifests/train_manifest.json"
VAL_MANIFEST     = "/kaggle/working/pcov-kws-manifests/val_manifest.json"
EMBEDDINGS_PATH  = "/kaggle/working/sv_embeddings.pt"
MUSAN_DIR        = "/kaggle/input/datasets/nhattruongdev/musan-noise"
RIR_DIR          = "/kaggle/input/datasets/nhattruongdev/rirs-noises"

CHECKPOINT_DIR   = "/kaggle/working/checkpoints"
BACKDOOR_PATH    = "/kaggle/working/checkpoints/last_checkpoint.pth"

if os.path.exists(BACKDOOR_PATH):
    RESUME_PATH = BACKDOOR_PATH
    print(f"🎯 Found local checkpoint weights at {BACKDOOR_PATH}!")
else:
    RESUME_PATH = f"{CHECKPOINT_DIR}/last_checkpoint.pth"

BEST_PATH    = f"{CHECKPOINT_DIR}/pcov_kws_best.pth"
NUM_EPOCHS   = 50
BATCH_SIZE   = 64
SV_FREEZE_EPOCHS = 15  # SV head frozen for first 15 epochs


# ─────────────────────────────────────────────
# 1. PCGRAD — GRADIENT SURGERY OPTIMIZER (FIXED)
# ─────────────────────────────────────────────
class PCGradOptimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, scaler=None):
        if scaler is not None:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

    def pc_backward(self, losses, scaler=None):
        num_tasks  = len(losses)
        task_grads = []

        for i in range(num_tasks):
            self.optimizer.zero_grad()
            retain = (i < num_tasks - 1)

            if scaler is not None:
                # Scale and backward — do NOT manually unscale here
                scaler.scale(losses[i]).backward(retain_graph=retain)
                grads = [
                    p.grad.detach().clone() if p.grad is not None
                    else torch.zeros_like(p)
                    for p in self.optimizer.param_groups[0]['params']
                ]
            else:
                losses[i].backward(retain_graph=retain)
                grads = [
                    p.grad.detach().clone() if p.grad is not None
                    else torch.zeros_like(p)
                    for p in self.optimizer.param_groups[0]['params']
                ]
            task_grads.append(grads)

        self.optimizer.zero_grad()

        # Gradient surgery — operates on scaled grads (consistent scale, surgery still valid)
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i == j:
                    continue
                dot = sum(
                    torch.sum(g_i * g_j)
                    for g_i, g_j in zip(task_grads[i], task_grads[j])
                )
                if dot < 0:
                    norm_sq = sum(
                        torch.sum(g_j * g_j) for g_j in task_grads[j]
                    ) + 1e-8
                    task_grads[i] = [
                        g_i - (dot / norm_sq) * g_j
                        for g_i, g_j in zip(task_grads[i], task_grads[j])
                    ]

        # Write summed gradients back — still scaled at this point
        for param_idx, p in enumerate(self.optimizer.param_groups[0]['params']):
            p.grad = sum(task_grads[i][param_idx] for i in range(num_tasks))


# ─────────────────────────────────────────────
# 2. DATASET WRAPPER
# ─────────────────────────────────────────────
class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, embeddings, offset=0):
        self.base   = base_dataset
        self.emb    = embeddings
        self.offset = offset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        waveform, kws_label, _ = self.base[idx]
        sv_emb = self.emb[self.offset + idx]
        return waveform, kws_label, sv_emb


# ─────────────────────────────────────────────
# 3. TRAINING EPOCH (FIXED)
# ─────────────────────────────────────────────
def train_epoch(model, frontend, dataloader,
                kws_hard_criterion, sv_kd_criterion,
                pc_optimizer, scaler, device, scheduler):

    model.train()
    frontend.train()

    total_kws_loss, total_sv_loss = 0.0, 0.0
    correct_kws, total_kws = 0, 0
    target_ones = torch.ones(BATCH_SIZE, device=device)

    for waveform, kws_label, t_sv_emb in tqdm(dataloader, desc="Training", leave=False):
        waveform  = waveform.to(device)
        kws_label = kws_label.to(device)
        t_sv_emb  = t_sv_emb.to(device)

        if waveform.size(0) != target_ones.size(0):
            target_ones = torch.ones(waveform.size(0), device=device)

        pc_optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            features            = frontend(waveform)
            s_sv_emb, s_kws_out = model(features)

        if kws_hard_criterion.weight is not None:
            kws_hard_criterion.weight = kws_hard_criterion.weight.to(s_kws_out.dtype)

        loss_kws = kws_hard_criterion(s_kws_out, kws_label)
        loss_sv  = sv_kd_criterion(
            s_sv_emb,
            t_sv_emb.to(s_sv_emb.dtype).detach(),
            hard_sv_loss=torch.tensor(0.0, device=device)
        )

        # 🛡️ NaN shield — no scheduler.step() on skip
        if torch.isnan(loss_kws) or torch.isnan(loss_sv) or \
                torch.isinf(loss_kws) or torch.isinf(loss_sv):
            print("\n⚠️ NaN/Inf loss detected — skipping batch...")
            pc_optimizer.zero_grad()
            continue

        # pc_backward writes scaled grads into p.grad
        pc_optimizer.pc_backward([loss_kws, loss_sv], scaler=scaler)

        # Unscale once here — converts scaled grads to true grads for clip
        if scaler is not None:
            scaler.unscale_(pc_optimizer.optimizer)

        # Clip on true gradient magnitudes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    
        torch.nn.utils.clip_grad_norm_(frontend.parameters(), max_norm=1.0)

        # scaler.step() checks for inf/nan, skips optimizer if found
        pc_optimizer.step(scaler=scaler)
        scheduler.step()

        total_kws_loss += loss_kws.item()
        total_sv_loss  += loss_sv.item()
        preds           = torch.argmax(s_kws_out, dim=1)
        correct_kws    += (preds == kws_label).sum().item()
        total_kws      += kws_label.size(0)

    n = len(dataloader)
    return total_kws_loss / n, total_sv_loss / n, 100.0 * correct_kws / total_kws

# ─────────────────────────────────────────────
# 4. VALIDATION EPOCH
# ─────────────────────────────────────────────
def validate_epoch(model, frontend, dataloader,
                   kws_hard_criterion, sv_kd_criterion,  # ← renamed
                   device):

    model.eval()
    frontend.eval()

    total_kws_loss, total_sv_loss = 0.0, 0.0
    correct_kws, total_kws = 0, 0
    target_ones = torch.ones(BATCH_SIZE, device=device)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for waveform, kws_label, t_sv_emb in tqdm(dataloader, desc="Validating", leave=False):
            waveform  = waveform.to(device)
            kws_label = kws_label.to(device)
            t_sv_emb  = t_sv_emb.to(device)

            if waveform.size(0) != target_ones.size(0):
                target_ones = torch.ones(waveform.size(0), device=device)

            with torch.amp.autocast('cuda'):
                features            = frontend(waveform)
                s_sv_emb, s_kws_out = model(features)

            if kws_hard_criterion.weight is not None:
                kws_hard_criterion.weight = kws_hard_criterion.weight.to(s_kws_out.dtype)

            loss_kws = kws_hard_criterion(s_kws_out, kws_label)
            loss_sv  = sv_kd_criterion(                      # ← SVEmbeddingKDLoss
                s_sv_emb,
                t_sv_emb.to(s_sv_emb.dtype),
                hard_sv_loss=torch.tensor(0.0, device=device)
            )

            total_kws_loss += loss_kws.item()
            total_sv_loss  += loss_sv.item()
            preds           = torch.argmax(s_kws_out, dim=1)
            correct_kws    += (preds == kws_label).sum().item()
            total_kws      += kws_label.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(kws_label.cpu().numpy())

    n = len(dataloader)
    target_names = ['Wake Word', 'Silence', 'Unknown', 'Hard-Negative']
    report = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        zero_division=0
    )

    return total_kws_loss / n, total_sv_loss / n, 100.0 * correct_kws / total_kws, report


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("🚀 PCOV-KWS PHASE 2 — 404NOT-FOUND TRAINING PIPELINE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Hardware: {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    with open(TRAIN_MANIFEST) as f:
        train_manifest = json.load(f)
    with open(VAL_MANIFEST) as f:
        val_manifest = json.load(f)
    print(f"📂 Train: {len(train_manifest):,} | Val: {len(val_manifest):,} samples")

    print(f"📦 Loading precomputed SV embeddings from {EMBEDDINGS_PATH}...")
    emb_data    = torch.load(EMBEDDINGS_PATH, map_location='cpu')
    embeddings  = emb_data['embeddings']
    train_count = emb_data['train_count']

    train_base = PCOVMultiTaskDataset(
        data_manifest=train_manifest,
        musan_dir=MUSAN_DIR,
        rir_dir=RIR_DIR
    )
    val_base = PCOVMultiTaskDataset(
        data_manifest=val_manifest,
        musan_dir=None,
        rir_dir=None
    )

    train_dataset = EmbeddingDataset(train_base, embeddings, offset=0)
    val_dataset   = EmbeddingDataset(val_base,   embeddings, offset=train_count)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    frontend = PCENFrontend().to(device)
    model    = PCOVKWS(num_kws_classes=4, sv_embedding_dim=192).to(device)

    # ── SV head freeze helper ──
    def set_sv_frozen(frozen: bool):
        for p in model.sv_head.parameters():
            p.requires_grad = not frozen
        print(f"{'🔒 SV head FROZEN' if frozen else '🔓 SV head UNFROZEN'}")

    set_sv_frozen(frozen=True)  # freeze at init

    print("📊 Computing class weights dynamically from train_manifest...")
    label_key = 'label' if 'label' in train_manifest[0] else 'kws_label'
    counts = [0, 0, 0, 0]
    for item in train_manifest:
        counts[item[label_key]] += 1
    counts = [max(c, 1) for c in counts]
    print(f"Detected Class Counts: {counts}")

    class_counts  = torch.tensor(counts, dtype=torch.float, device=device)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 4.0

    kws_hard_criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    sv_kd_criterion    = SVEmbeddingKDLoss(alpha=0.5).to(device)


    base_optimizer = optim.AdamW(
        list(model.parameters()) + list(frontend.parameters()),
        lr=5e-4, weight_decay=1e-4
    )

    pc_optimizer = PCGradOptimizer(base_optimizer) 
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        base_optimizer,
        max_lr=5e-4,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )


    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    start_epoch   = 0
    best_val_loss = float('inf')

    if os.path.exists(RESUME_PATH):
        print(f"🔄 Resuming from {RESUME_PATH}")
        ckpt = torch.load(RESUME_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        frontend.load_state_dict(ckpt['frontend_state_dict'])
        base_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                print("✅ Scheduler state restored")
            except Exception as e:
                print(f"⚠️ Scheduler state mismatch — starting scheduler fresh: {e}")
        if scaler and ckpt.get('scaler_state_dict'):
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"⏩ Resuming from Epoch {start_epoch + 1}/{NUM_EPOCHS}")

        # Restore freeze state on resume
        if start_epoch < SV_FREEZE_EPOCHS:
            set_sv_frozen(frozen=True)
        else:
            set_sv_frozen(frozen=False)
    else:
        print("🆕 No checkpoint — starting fresh")

    print(f"\n✅ Pipeline ready. Training on {device}...\n")

    for epoch in range(start_epoch, NUM_EPOCHS):

        # ── SV freeze/unfreeze boundary ──
        if epoch == SV_FREEZE_EPOCHS:
            set_sv_frozen(frozen=False)

        print(f"--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        train_kws_l, train_sv_l, train_acc = train_epoch(
            model, frontend, train_loader,
            kws_hard_criterion, sv_kd_criterion,
            pc_optimizer, scaler, device, scheduler
        )

        val_kws_l, val_sv_l, val_acc, val_report = validate_epoch(
            model, frontend, val_loader,
            kws_hard_criterion, sv_kd_criterion,
            device
        )

        combined_val_loss = 0.5 * val_kws_l + 0.5 * val_sv_l

        print(f"Train | KWS: {train_kws_l:.4f} | SV: {train_sv_l:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val   | KWS: {val_kws_l:.4f}   | SV: {val_sv_l:.4f}   | Acc: {val_acc:.2f}%")
        print(f"LR    | {scheduler.get_last_lr()[0]:.6f}")

        print("\n" + "-" * 50)
        print("📊 VALIDATION CLASSIFICATION REPORT")
        print("-" * 50)
        print(val_report)
        print("-" * 50 + "\n")

        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'frontend_state_dict':  frontend.state_dict(),
            'optimizer_state_dict': base_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict':    scaler.state_dict() if scaler else None,
            'val_loss':             best_val_loss,
        }, RESUME_PATH)

        if combined_val_loss < best_val_loss:
            best_val_loss = combined_val_loss
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'frontend_state_dict': frontend.state_dict(),
                'val_loss':            best_val_loss,
            }, BEST_PATH)
            print(f"🌟 New best saved! Val Loss: {best_val_loss:.4f}")

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print(f"📂 Best model: {BEST_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()