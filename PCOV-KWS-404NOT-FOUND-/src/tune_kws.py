import os, json, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from frontend import PCENFrontend
from model import PCOVKWS
from dataset import PCOVMultiTaskDataset

MANI_DIR = "/kaggle/working/pcov-kws-manifests"
FT_TRAIN = f"{MANI_DIR}/finetune_train_manifest.json"
FT_VAL   = f"{MANI_DIR}/finetune_val_manifest.json"
HOLDOUT  = f"{MANI_DIR}/real_holdout_manifest.json"
NEG_GATE = f"{MANI_DIR}/neg_gate_manifest.json"          # ← NEW

CKPT_DIR = "/kaggle/working/checkpoints"
BASE_CKPT = f"{CKPT_DIR}/pcov_kws_best.pth"
FT_CKPT   = f"{CKPT_DIR}/pcov_kws_finetuned.pth"

MUSAN_DIR = "/kaggle/input/datasets/nhattruongdev/musan-noise"
RIR_DIR   = "/kaggle/input/datasets/nhattruongdev/rirs-noises"

MAX_EPOCHS = 15; BATCH_SIZE = 64; LR = 1.5e-5; WEIGHT_DECAY = 1e-4
WW_HARD_GATE = 0.95; NEG_COLLAPSE_GATE = 0.50
LABELS = ['WakeWord','Silence','Unknown','HardNeg']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def configure_trainable(model):
    for p in model.parameters(): p.requires_grad = False
    for p in model.kws_head.parameters(): p.requires_grad = True
    for blk in ["block3","block4"]:
        for m in getattr(model,blk).modules():
            if isinstance(m,(nn.Conv2d,nn.BatchNorm2d)):
                for p in m.parameters(): p.requires_grad = True
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Trainable: {nt:,}")

def set_bn_mode(model):
    model.eval(); model.block3.train(); model.block4.train(); model.kws_head.train()

def make_loader(path, shuffle, use_aug):
    with open(path) as f: man = json.load(f)
    ds = PCOVMultiTaskDataset(
        man,
        musan_dir=(MUSAN_DIR if use_aug else None),   # ← noise ONLY on train
        rir_dir=(RIR_DIR if use_aug else None),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2, pin_memory=True)

@torch.no_grad()
def per_class_recall(model, frontend, loader):
    model.eval(); frontend.eval()
    preds, labels = [], []
    for wav, kl, _ in loader:
        wav = wav.to(device)
        _, logits = model(frontend(wav))
        preds += torch.argmax(logits,1).cpu().tolist(); labels += kl.tolist()
    rec = {}
    for c in range(4):
        tot = sum(1 for l in labels if l==c)
        cor = sum(1 for p,l in zip(preds,labels) if l==c and p==c)
        rec[c] = cor/tot if tot>0 else None
    return rec

@torch.no_grad()
def holdout_fire_rate(model, frontend, loader):   # real WW SHOULD fire class 0
    model.eval(); frontend.eval(); fired = total = 0
    for wav, _, _ in loader:
        wav = wav.to(device)
        _, logits = model(frontend(wav))
        p = torch.argmax(logits,1).cpu()
        fired += (p==0).sum().item(); total += p.numel()
    return fired/max(total,1)

@torch.no_grad()
def negative_rejection_rate(model, frontend, loader):   # ← YOUR ADDITION
    # fraction of known NON-wake clips that are NOT classified as WakeWord
    model.eval(); frontend.eval(); rejected = total = 0
    for wav, _, _ in loader:
        wav = wav.to(device)
        _, logits = model(frontend(wav))
        p = torch.argmax(logits,1).cpu()
        rejected += (p!=0).sum().item(); total += p.numel()
    return rejected/max(total,1)

def main():
    frontend = PCENFrontend().to(device)
    model = PCOVKWS(num_kws_classes=4, sv_embedding_dim=192).to(device)
    ckpt = torch.load(BASE_CKPT, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    frontend.load_state_dict(ckpt['frontend_state_dict']); frontend.eval()
    for p in frontend.parameters(): p.requires_grad = False
    configure_trainable(model)

    train_loader   = make_loader(FT_TRAIN, True,  use_aug=True)    # noise ON
    val_loader     = make_loader(FT_VAL,   False, use_aug=False)   # clean
    holdout_loader = make_loader(HOLDOUT,  False, use_aug=False)   # clean real
    neg_loader     = make_loader(NEG_GATE, False, use_aug=False)   # clean negs

    counts = [0,0,0,0]
    with open(FT_TRAIN) as f:
        for it in json.load(f): counts[it['kws_label']] += 1
    counts = [max(c,1) for c in counts]
    cw = torch.tensor([1.0/c for c in counts], device=device); cw = cw/cw.sum()*4.0
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=WEIGHT_DECAY)

    best = -1.0
    for epoch in range(1, MAX_EPOCHS+1):
        set_bn_mode(model); running = 0.0
        for wav, kl, _ in train_loader:
            wav, kl = wav.to(device), kl.to(device)
            optimizer.zero_grad()
            with torch.no_grad(): feats = frontend(wav)
            _, logits = model(feats)
            loss = criterion(logits, kl)
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); running += loss.item()

        rec = per_class_recall(model, frontend, val_loader)
        hold = holdout_fire_rate(model, frontend, holdout_loader)
        neg_rej = negative_rejection_rate(model, frontend, neg_loader)   # ← NEW

        rstr = " | ".join(f"{LABELS[c]}:{(rec[c]*100 if rec[c] is not None else -1):.0f}%"
                          for c in range(4))
        print(f"[E{epoch}] loss={running/len(train_loader):.4f} | "
              f"real_fire={hold*100:.0f}% | neg_reject={neg_rej*100:.0f}% | synth: {rstr}")

        synth_ww = rec[0] or 0.0
        collapsed = any(rec[c] is not None and rec[c] < NEG_COLLAPSE_GATE*0.90 for c in [1,2,3])
        if collapsed or synth_ww < 0.985:        # ← was WW_HARD_GATE*0.95 (=0.9025)
            continue


        kpi_health = min([rec[c]/0.95 for c in range(4) if rec[c] is not None])
        # ← YOUR ADDITION: real must fire AND negatives must be rejected
        score = hold * kpi_health * neg_rej
        if score > best:
            best = score
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'frontend_state_dict': frontend.state_dict()}, FT_CKPT)
            print(f"  🌟 Saved best (score={score:.3f}, real_fire={hold*100:.0f}%, "
                  f"neg_reject={neg_rej*100:.0f}%)")

if __name__ == "__main__":
    main()
