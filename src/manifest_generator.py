import os
import json
import pandas as pd
import glob
import random
from collections import Counter

# ─────────────────────────────────────────────
# 1. KAGGLE DATASET PATHS
# ─────────────────────────────────────────────
COQUI_DIR = "/kaggle/input/datasets/abhishekkujur19/coqui-dataset"
GSC_DIR   = "/kaggle/input/datasets/sylkaladin/speech-commands-v2"
LIBRIPHRASE_DIR = "/kaggle/working/libriphrase_raw"
OUTPUT_DIR = "/kaggle/working/pcov-kws-manifests"

os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAIN_JSON_PATH = os.path.join(OUTPUT_DIR, 'train_manifest.json')
VAL_JSON_PATH   = os.path.join(OUTPUT_DIR, 'val_manifest.json')

train_manifest = []
val_manifest   = []
random.seed(42)  # Ensures fully reproducible splits


# ─────────────────────────────────────────────
# 1. COQUI SYNTHETIC DATA (CLASSES 0 & 3)
# ─────────────────────────────────────────────
print("--- TASK 1: PROCESSING COQUI SYNTHETIC DATA ---")

coqui_csv = os.path.join(COQUI_DIR, 'manifest.csv')
if not os.path.exists(coqui_csv):
    coqui_csv = 'manifest.csv'
    if not os.path.exists(coqui_csv):
        raise FileNotFoundError("❌ Could not locate Coqui manifest.csv!")

coqui_df = pd.read_csv(coqui_csv)

# Strict Speaker Isolation Split
# Train and val see completely different speakers — tests real generalization
all_speakers = coqui_df['speaker'].unique().tolist()
random.shuffle(all_speakers)
split_idx = int(len(all_speakers) * 0.8)

train_speakers = set(all_speakers[:split_idx])
val_speakers   = set(all_speakers[split_idx:])

print(f"Total Unique Synthetic Speakers: {len(all_speakers)}")
print(f"├── Train Speakers: {len(train_speakers)}")
print(f"└── Val Speakers:   {len(val_speakers)} (strictly unseen voices)")

coqui_train_count = 0
coqui_val_count   = 0

for _, row in coqui_df.iterrows():
    # [FIX 1] Specific exception types — no bare except
    try:
        spk_id = int(str(row['speaker']).replace('p', ''))
    except (ValueError, AttributeError):
        spk_id = -1

    sample = {
        'path':      os.path.join(COQUI_DIR, row['filename']),
        'kws_label': int(row['label']),  # 0 = Wake Word, 3 = Hard Negative
        'spk_label': spk_id
    }

    if row['speaker'] in train_speakers:
        train_manifest.append(sample)
        coqui_train_count += 1
    else:
        val_manifest.append(sample)
        coqui_val_count += 1

print(f"✅ Coqui Complete:")
print(f"├── Train: {coqui_train_count} tracks")
print(f"└── Val:   {coqui_val_count} tracks")


# ─────────────────────────────────────────────
# 2. LIBRIPHRASE SPEECH NEGATIVES (CLASS 2)
# ─────────────────────────────────────────────
print("\n--- TASK 2: PROCESSING LIBRIPHRASE SPEECH NEGATIVES ---")

libri_wavs = glob.glob(os.path.join(LIBRIPHRASE_DIR, "**/*.wav"), recursive=True)

if libri_wavs:
    random.shuffle(libri_wavs)
    libri_split = int(len(libri_wavs) * 0.8)

    # [FIX 3] Slice pattern — cleaner, no index tracking
    for wav_path in libri_wavs[:libri_split]:
        train_manifest.append({'path': wav_path, 'kws_label': 2, 'spk_label': -1})
    for wav_path in libri_wavs[libri_split:]:
        val_manifest.append({'path': wav_path, 'kws_label': 2, 'spk_label': -1})

    print(f"✅ LibriPhrase Complete:")
    print(f"├── Train: {libri_split} tracks")
    print(f"└── Val:   {len(libri_wavs) - libri_split} tracks")
else:
    print("⚠️ No LibriPhrase .wav files found. Skipping.")


# ─────────────────────────────────────────────
# 3. GOOGLE SPEECH COMMANDS V2 (CLASSES 1 & 2)
# ─────────────────────────────────────────────
print("\n--- TASK 3: PROCESSING GOOGLE SPEECH COMMANDS V2 ---")

if os.path.exists(GSC_DIR):
    gsc_wavs = glob.glob(os.path.join(GSC_DIR, "**/*.wav"), recursive=True)
    random.shuffle(gsc_wavs)
    
    gsc_train_words = 0
    gsc_val_words = 0
    silence_train_count = 0
    silence_val_count = 0
    
    # 80/20 split index for the raw files
    gsc_split = int(len(gsc_wavs) * 0.8)

    # Process Training Split
    for wav_path in gsc_wavs[:gsc_split]:
        if "_background_noise_" in wav_path or "silence" in wav_path.lower():
            # Oversample Silence for Training (2000 crops per file)
            sample = {'path': wav_path, 'kws_label': 1, 'spk_label': -1}
            train_manifest.extend([sample] * 2000)
            silence_train_count += 2000
        else:
            # Standard Unknown Word (Class 2)
            train_manifest.append({'path': wav_path, 'kws_label': 2, 'spk_label': -1})
            gsc_train_words += 1

    # Process Validation Split
    for wav_path in gsc_wavs[gsc_split:]:
        if "_background_noise_" in wav_path or "silence" in wav_path.lower():
            # Oversample Silence for Validation (500 crops per file)
            sample = {'path': wav_path, 'kws_label': 1, 'spk_label': -1}
            val_manifest.extend([sample] * 500)
            silence_val_count += 500
        else:
            # Standard Unknown Word (Class 2)
            val_manifest.append({'path': wav_path, 'kws_label': 2, 'spk_label': -1})
            gsc_val_words += 1

    print(f"✅ Google Speech Commands V2 Complete:")
    print(f"├── Train: {gsc_train_words} words | {silence_train_count} silence crops")
    print(f"└── Val:   {gsc_val_words} words | {silence_val_count} silence crops")
else:
    print("⚠️ Google Speech Commands directory not found. Skipping.")

# ─────────────────────────────────────────────
# 4. SHUFFLE FINAL MANIFESTS
# ─────────────────────────────────────────────
# Shuffle so batches contain mixed classes — prevents loss spikes
random.shuffle(train_manifest)
random.shuffle(val_manifest)


# ─────────────────────────────────────────────
# 5. EXPORT TO JSON
# ─────────────────────────────────────────────
print("\n--- EXPORTING MANIFESTS ---")

with open(TRAIN_JSON_PATH, 'w') as f:
    json.dump(train_manifest, f, indent=4)

with open(VAL_JSON_PATH, 'w') as f:
    json.dump(val_manifest, f, indent=4)

print(f"✅ Train manifest → {TRAIN_JSON_PATH}")
print(f"✅ Val   manifest → {VAL_JSON_PATH}")


# ─────────────────────────────────────────────
# 6. CLASS DISTRIBUTION REPORT
# ─────────────────────────────────────────────
# [FIX 4] Catch class imbalance before training starts
print("\n--- CLASS DISTRIBUTION REPORT ---")
CLASS_NAMES = {0: 'Wake Word', 1: 'Silence', 2: 'Unknown Speech', 3: 'Hard Negative'}

train_labels = Counter(s['kws_label'] for s in train_manifest)
val_labels   = Counter(s['kws_label'] for s in val_manifest)

print("Train split:")
for lbl in sorted(train_labels):
    print(f"  Class {lbl} ({CLASS_NAMES[lbl]:>15}): {train_labels[lbl]:>7,} samples")

print("Val split:")
for lbl in sorted(val_labels):
    print(f"  Class {lbl} ({CLASS_NAMES[lbl]:>15}): {val_labels[lbl]:>7,} samples")

print("\n" + "=" * 55)
print(f"🚀 TRAIN TOTAL : {len(train_manifest):,} samples")
print(f"🎯 VAL TOTAL   : {len(val_manifest):,} samples")
print("=" * 55)