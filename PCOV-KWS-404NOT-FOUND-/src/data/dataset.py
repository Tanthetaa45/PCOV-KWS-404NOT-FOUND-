import os
import random
import glob
import torch
import torchaudio.functional as F_audio
from torch.utils.data import Dataset
import soundfile as sf
from tqdm import tqdm

def load_audio_safe(file_path):
    clean_path = str(file_path).replace('\n', '').replace('\r', '').strip()
    wav, sr = sf.read(clean_path)
    waveform = torch.from_numpy(wav).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.t()
    return waveform, sr


# ─────────────────────────────────────────────
# 1. RAM-CACHED RIR (Zero Disk I/O)
# ─────────────────────────────────────────────
class RAMCachedRIR:
    def __init__(self, rir_dir, sample_rate=16000, max_cache=2000):
        self.sample_rate = sample_rate
        all_files = glob.glob(os.path.join(rir_dir, '**', '*.wav'), recursive=True)
        self.cache = []

        if not all_files:
            print("⚠️ No RIR files found. Distance simulation will be bypassed.")
            return

        random.shuffle(all_files)
        selected_files = all_files[:max_cache]

        print(f"📥 Caching {len(selected_files)} RIR files into RAM...")
        for path in tqdm(selected_files, desc="Loading RIRs", leave=False):
            wav, sr = load_audio_safe(path)
            if sr != sample_rate:
                wav = F_audio.resample(wav, sr, sample_rate)
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            max_len = sample_rate * 2
            if wav.shape[1] > max_len:
                wav = wav[:, :max_len]
            norm = torch.norm(wav, p=2)
            if norm > 0:
                wav = wav / norm
            self.cache.append(wav)

    def __call__(self, clean_waveform):
        if not self.cache:
            return clean_waveform
        rir_waveform = random.choice(self.cache)
        augmented = F_audio.fftconvolve(clean_waveform, rir_waveform)
        return torch.clamp(augmented[:, :clean_waveform.shape[1]], min=-1.0, max=1.0)


# ─────────────────────────────────────────────
# 2. RAM-CACHED MUSAN (Zero Disk I/O)
# ─────────────────────────────────────────────
class RAMCachedMUSAN:
    def __init__(self, musan_dir, sample_rate=16000, max_cache=2000):
        self.sample_rate = sample_rate
        all_files = glob.glob(os.path.join(musan_dir, '**', '*.wav'), recursive=True)
        self.cache = []

        if not all_files:
            print("⚠️ No MUSAN files found. Noise augmentation will be bypassed.")
            return

        random.shuffle(all_files)
        selected_files = all_files[:max_cache]

        print(f"📥 Caching {len(selected_files)} MUSAN files into RAM...")
        for path in tqdm(selected_files, desc="Loading MUSAN", leave=False):
            wav, sr = load_audio_safe(path)
            if sr != sample_rate:
                wav = F_audio.resample(wav, sr, sample_rate)
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            max_len = sample_rate * 5
            if wav.shape[1] > max_len:
                start = random.randint(0, wav.shape[1] - max_len)
                wav = wav[:, start:start+max_len]
            self.cache.append(wav)

    def __call__(self, clean_waveform, target_snr_db=-5.0):
        if not self.cache:
            return clean_waveform
        noise_waveform = random.choice(self.cache)
        if noise_waveform.shape[1] > clean_waveform.shape[1]:
            start = random.randint(0, noise_waveform.shape[1] - clean_waveform.shape[1])
            noise_waveform = noise_waveform[:, start:start + clean_waveform.shape[1]]
        else:
            pad_amount = clean_waveform.shape[1] - noise_waveform.shape[1]
            noise_waveform = torch.nn.functional.pad(noise_waveform, (0, pad_amount))
        clean_power = torch.mean(clean_waveform ** 2)
        noise_power = torch.mean(noise_waveform ** 2)
        if clean_power == 0 or noise_power == 0:
            return clean_waveform
        k = torch.sqrt((clean_power / noise_power) * (10 ** (-target_snr_db / 10)))
        noisy = clean_waveform + k * noise_waveform
        return torch.clamp(noisy, min=-1.0, max=1.0)


# ─────────────────────────────────────────────
# 3. THE MASTER MULTI-TASK DATASET
# ─────────────────────────────────────────────
class PCOVMultiTaskDataset(Dataset):
    def __init__(self, data_manifest, musan_dir=None, rir_dir=None,
                 sample_rate=16000, duration_sec=1.0):
        self.data_list     = data_manifest
        self.sample_rate   = sample_rate
        self.target_length = int(sample_rate * duration_sec)
        self.rir_augmenter   = RAMCachedRIR(rir_dir, sample_rate)     if rir_dir   else None
        self.musan_augmenter = RAMCachedMUSAN(musan_dir, sample_rate) if musan_dir else None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        waveform, sr = load_audio_safe(item['path'])

        # ── Resample + mono FIRST ──
        if sr != self.sample_rate:
            waveform = F_audio.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # ── NaN/silence guard — local vars only, never mutate item dict ──
        if torch.isnan(waveform).any() or torch.isinf(waveform).any() \
                or waveform.pow(2).mean().sqrt() < 1e-8:
            waveform  = torch.zeros(1, self.target_length)
            kws_label = 1   # silence
            spk_label = -1
        else:
            kws_label = item['kws_label']
            spk_label = item.get('spk_label', -1)

        # ── Crop / pad ──
        current_len = waveform.shape[1]
        chunk_idx   = item.get('chunk_idx', None)

        if chunk_idx is not None:
            start = chunk_idx * self.target_length
            end   = start + self.target_length
            if end <= current_len:
                waveform = waveform[:, start:end]
            else:
                waveform = waveform[:, start:] if start < current_len \
                           else torch.zeros(1, self.target_length)
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.target_length - waveform.shape[1])
                )
        elif current_len > self.target_length:
            if kws_label in [0, 3]:
                start = 0
            else:
                start = random.randint(0, current_len - self.target_length)
            waveform = waveform[:, start:start + self.target_length]
        elif current_len < self.target_length:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.target_length - current_len)
            )

        # ── Augmentations ──
        if self.rir_augmenter and random.random() < 0.30:
            waveform = self.rir_augmenter(waveform)
        if self.musan_augmenter and random.random() < 0.50:
            snr = random.uniform(-5.0, 0.0) if random.random() < 0.40 \
                  else random.uniform(0.0, 10.0)
            waveform = self.musan_augmenter(waveform, target_snr_db=snr)

        peak = waveform.abs().max()
        if peak > 1e-4:
            waveform = waveform / peak * 0.9

        return (
            waveform,
            torch.tensor(kws_label, dtype=torch.long),
            torch.tensor(spk_label, dtype=torch.long),
        )