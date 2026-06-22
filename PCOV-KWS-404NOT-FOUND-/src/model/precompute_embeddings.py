
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import soundfile as sf
import torchaudio.functional as F_audio
from teacher_loader import ECAPATeacher

TRAIN_MANIFEST = "/kaggle/working/pcov-kws-manifests/train_manifest.json"
VAL_MANIFEST   = "/kaggle/working/pcov-kws-manifests/val_manifest.json"
OUTPUT_PATH    = "/kaggle/working/sv_embeddings.pt"
BATCH_SIZE     = 256
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RawAudioDataset(Dataset):
    def __init__(self, manifest):
        self.manifest   = manifest
        self.target_len = 16000

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        raw_path = self.manifest[idx]['path']
        path = str(raw_path).replace('\n', '').replace('\r', '').strip()
        try:
            wav, sr  = sf.read(path)
            waveform = torch.from_numpy(wav).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()
            if sr != 16000:
                waveform = F_audio.resample(waveform, sr, 16000)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if waveform.shape[1] > self.target_len:
                waveform = waveform[:, :self.target_len]
            elif waveform.shape[1] < self.target_len:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.target_len - waveform.shape[1])
                )
        except Exception as e:
            print(f"⚠️ Failed: {path} — {e}")
            waveform = torch.zeros(1, self.target_len)

        # 🛡️ THE VACCINE: Catch NaN, Inf, or ultra-low RMS instantly!
        if torch.isnan(waveform).any() or torch.isinf(waveform).any() or waveform.pow(2).mean().sqrt() < 1e-8:
            waveform = torch.zeros(1, self.target_len)

        return waveform, idx

def main():
    print("=" * 60)
    print("🚀 ECAPA-TDNN EMBEDDING PRE-COMPUTATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    with open(TRAIN_MANIFEST) as f:
        train_manifest = json.load(f)
    with open(VAL_MANIFEST) as f:
        val_manifest = json.load(f)

    full_manifest = train_manifest + val_manifest
    total         = len(full_manifest)
    print(f"Total files: {total:,}")

    sv_teacher = ECAPATeacher(embedding_dim=192).to(DEVICE)
    for p in sv_teacher.parameters():
        p.requires_grad = False
    sv_teacher.eval()
    if sv_teacher.use_real_model:
        sv_teacher.sb_model.device = DEVICE
        sv_teacher.sb_model.mods   = sv_teacher.sb_model.mods.to(DEVICE)
    print("✅ ECAPA-TDNN loaded and frozen")

    all_embeddings = torch.zeros(total, 192, dtype=torch.float32)

    dataset = RawAudioDataset(full_manifest)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"\nProcessing {total:,} files...")
    with torch.no_grad():
        for waveforms, indices in tqdm(loader, desc="Pre-computing"):
            waveforms  = waveforms.to(DEVICE)
            embeddings = sv_teacher(waveforms).cpu().float()
            
            # 🛡️ THE SECOND VACCINE: Catch any NaN embeddings
            nan_mask = torch.isnan(embeddings).any(dim=1)
            if nan_mask.any():
                safe_vec = torch.zeros(192, dtype=torch.float32)
                safe_vec[0] = 1.0          # unit vector — valid on hypersphere
                embeddings[nan_mask] = safe_vec
                print(f"\n⚠️ Fixed {nan_mask.sum().item()} NaN embeddings in batch!")

            all_embeddings[indices] = embeddings

    torch.save({
        'train_count': len(train_manifest),
        'val_count':   len(val_manifest),
        'embeddings':  all_embeddings,
    }, OUTPUT_PATH)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\n✅ Saved {total:,} embeddings → {OUTPUT_PATH}")
    print(f"   Size: {size_mb:.1f} MB | Shape: {all_embeddings.shape}")
    print("✅ Done. Now run train.py")

if __name__ == "__main__":
    main()