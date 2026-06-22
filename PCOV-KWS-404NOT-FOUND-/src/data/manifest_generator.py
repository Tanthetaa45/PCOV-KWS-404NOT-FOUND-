import os
import json
import pandas as pd
import glob
import random
from collections import Counter

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
COQUI_DIR       = "/kaggle/working/coqui_dataset"
GSC_DIR         = "/kaggle/input/datasets/sylkaladin/speech-commands-v2"
LIBRIPHRASE_DIR = "/kaggle/working/libriphrase_raw"
OUTPUT_DIR      = "/kaggle/working/pcov-kws-manifests"

os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAIN_JSON_PATH = os.path.join(OUTPUT_DIR, 'train_manifest.json')
VAL_JSON_PATH   = os.path.join(OUTPUT_DIR, 'val_manifest.json')

train_manifest = []
val_manifest   = []
random.seed(42)

# ─────────────────────────────────────────────
# TARGETS — tune these knobs
# ─────────────────────────────────────────────
TARGET_UNKNOWN_TRAIN = 40_000   # cap GSC unknown
TARGET_UNKNOWN_VAL   = 10_000
TARGET_SILENCE_TRAIN = 8_000    # silence crops (real diversity via chunking)
TARGET_SILENCE_VAL   = 2_000
LIBRI_CAP_TRAIN      = 20_000   # cap LibriPhrase unknown
LIBRI_CAP_VAL        = 5_000


# ─────────────────────────────────────────────
# 1. COQUI SYNTHETIC (Classes 0 & 3) — unchanged logic, just report
# ─────────────────────────────────────────────
print("--- TASK 1: COQUI SYNTHETIC DATA ---")

coqui_csv = os.path.join(COQUI_DIR, 'manifest.csv')
coqui_df  = pd.read_csv(coqui_csv)

all_speakers = coqui_df['speaker'].unique().tolist()
random.shuffle(all_speakers)
split_idx      = int(len(all_speakers) * 0.8)
train_speakers = set(all_speakers[:split_idx])
val_speakers   = set(all_speakers[split_idx:])

print(f"Speakers — Train: {len(train_speakers)} | Val: {len(val_speakers)}")

ww_train, ww_val, hn_train, hn_val = 0, 0, 0, 0
for _, row in coqui_df.iterrows():
    try:
        spk_id = int(str(row['speaker']).replace('p', ''))
    except (ValueError, AttributeError):
        spk_id = -1

    sample = {
        'path':      os.path.join(COQUI_DIR, row['filename']),
        'kws_label': int(row['label']),
        'spk_label': spk_id
    }
    if row['speaker'] in train_speakers:
        train_manifest.append(sample)
        if int(row['label']) == 0: ww_train += 1
        else: hn_train += 1
    else:
        val_manifest.append(sample)
        if int(row['label']) == 0: ww_val += 1
        else: hn_val += 1

print(f"Wake Word  — Train: {ww_train:,} | Val: {ww_val:,}")
print(f"Hard-Neg   — Train: {hn_train:,} | Val: {hn_val:,}")


# ─────────────────────────────────────────────
# 2. LIBRIPHRASE UNKNOWN (Class 2) — CAPPED
# ─────────────────────────────────────────────
print("\n--- TASK 2: LIBRIPHRASE UNKNOWN ---")

libri_wavs = glob.glob(os.path.join(LIBRIPHRASE_DIR, "**/*.wav"), recursive=True)
if libri_wavs:
    random.shuffle(libri_wavs)
    # Cap total before splitting so val cap is proportional
    libri_wavs = libri_wavs[:LIBRI_CAP_TRAIN + LIBRI_CAP_VAL]
    libri_split = int(len(libri_wavs) * 0.8)

    for wav_path in libri_wavs[:libri_split]:
        train_manifest.append({'path': wav_path, 'kws_label': 2, 'spk_label': -1})
    for wav_path in libri_wavs[libri_split:]:
        val_manifest.append({'path': wav_path, 'kws_label': 2, 'spk_label': -1})

    print(f"Train: {libri_split:,} | Val: {len(libri_wavs)-libri_split:,}")
else:
    print("⚠️ No LibriPhrase files found.")


# ─────────────────────────────────────────────
# 3. GSC V2 — CAPPED UNKNOWN + CHUNKED SILENCE
# ─────────────────────────────────────────────
print("\n--- TASK 3: GOOGLE SPEECH COMMANDS V2 ---")

if os.path.exists(GSC_DIR):
    all_gsc = glob.glob(os.path.join(GSC_DIR, "**/*.wav"), recursive=True)
    random.shuffle(all_gsc)

    silence_files = [p for p in all_gsc
                     if "_background_noise_" in p or "silence" in p.lower()]
    word_files    = [p for p in all_gsc
                     if "_background_noise_" not in p and "silence" not in p.lower()]

    # ── Unknown words: cap total, then 80/20 split ──
    word_files = word_files[:TARGET_UNKNOWN_TRAIN + TARGET_UNKNOWN_VAL]
    word_split = int(len(word_files) * 0.8)
    for p in word_files[:word_split]:
        train_manifest.append({'path': p, 'kws_label': 2, 'spk_label': -1})
    for p in word_files[word_split:]:
        val_manifest.append({'path': p, 'kws_label': 2, 'spk_label': -1})

    # ── Silence: chunk each file into non-overlapping 1s windows ──
    # This gives REAL diversity instead of 2000 copies of the same file
    SAMPLE_RATE   = 16000

# make_manifest.py — replace the silence chunking section only

    silence_chunks_train = []
    silence_chunks_val   = []

    import soundfile as sf_probe
    for sil_path in silence_files:
        try:
            info       = sf_probe.info(sil_path)
            total_secs = info.duration
            n_chunks   = int(total_secs)
            for chunk_idx in range(n_chunks):
                entry = {
                    'path':      sil_path,
                    'kws_label': 1,
                    'spk_label': -1,
                    'chunk_idx': chunk_idx
                }
                if random.random() < 0.8:
                    silence_chunks_train.append(entry)
                else:
                    silence_chunks_val.append(entry)
        except Exception:
            continue

    # ── If not enough silence chunks, repeat with random chunk_idx offsets ──
    if len(silence_chunks_train) < TARGET_SILENCE_TRAIN and silence_files:
        print(f"⚠️  Only {len(silence_chunks_train)} silence chunks — padding via random re-crops...")
        while len(silence_chunks_train) < TARGET_SILENCE_TRAIN:
            sil_path = random.choice(silence_files)
            try:
                info    = sf_probe.info(sil_path)
                n_chunks = max(1, int(info.duration))
                silence_chunks_train.append({
                    'path':      sil_path,
                    'kws_label': 1,
                    'spk_label': -1,
                    'chunk_idx': random.randint(0, n_chunks - 1)
                })
            except Exception:
                continue

    if len(silence_chunks_val) < TARGET_SILENCE_VAL and silence_files:
        while len(silence_chunks_val) < TARGET_SILENCE_VAL:
            sil_path = random.choice(silence_files)
            try:
                info    = sf_probe.info(sil_path)
                n_chunks = max(1, int(info.duration))
                silence_chunks_val.append({
                    'path':      sil_path,
                    'kws_label': 1,
                    'spk_label': -1,
                    'chunk_idx': random.randint(0, n_chunks - 1)
                })
            except Exception:
                continue

    # Cap to targets
    random.shuffle(silence_chunks_train)
    random.shuffle(silence_chunks_val)
    silence_chunks_train = silence_chunks_train[:TARGET_SILENCE_TRAIN]
    silence_chunks_val   = silence_chunks_val[:TARGET_SILENCE_VAL]

    train_manifest.extend(silence_chunks_train)
    val_manifest.extend(silence_chunks_val)

    print(f"Unknown  — Train: {word_split:,} | Val: {len(word_files)-word_split:,}")
    print(f"Silence  — Train: {len(silence_chunks_train):,} | Val: {len(silence_chunks_val):,}")
else:
    print("⚠️ GSC directory not found.")


# ─────────────────────────────────────────────
# 4. SHUFFLE + EXPORT
# ─────────────────────────────────────────────
random.shuffle(train_manifest)
random.shuffle(val_manifest)

with open(TRAIN_JSON_PATH, 'w') as f:
    json.dump(train_manifest, f, indent=4)
with open(VAL_JSON_PATH, 'w') as f:
    json.dump(val_manifest, f, indent=4)


# ─────────────────────────────────────────────
# 5. DISTRIBUTION REPORT
# ─────────────────────────────────────────────
CLASS_NAMES = {0: 'Wake Word', 1: 'Silence', 2: 'Unknown', 3: 'Hard-Negative'}
train_labels = Counter(s['kws_label'] for s in train_manifest)
val_labels   = Counter(s['kws_label'] for s in val_manifest)

print("\n--- CLASS DISTRIBUTION REPORT ---")
print(f"{'Class':<20} {'Train':>10} {'Train%':>8} {'Val':>10} {'Val%':>8}")
print("-" * 60)
total_train = len(train_manifest)
total_val   = len(val_manifest)
for lbl in sorted(CLASS_NAMES):
    tr = train_labels.get(lbl, 0)
    vl = val_labels.get(lbl, 0)
    print(f"{CLASS_NAMES[lbl]:<20} {tr:>10,} {100*tr/total_train:>7.1f}% "
          f"{vl:>10,} {100*vl/total_val:>7.1f}%")
print("-" * 60)
print(f"{'TOTAL':<20} {total_train:>10,} {'100.0%':>8} {total_val:>10,} {'100.0%':>8}")

# Imbalance warning
max_tr = max(train_labels.values())
min_tr = min(train_labels.values())
ratio  = max_tr / max(min_tr, 1)
if ratio > 10:
    print(f"\n⚠️  WARNING: Max/min class ratio = {ratio:.1f}x — consider generating more Hard-Neg/Wake Word data")
else:
    print(f"\n✅ Class ratio = {ratio:.1f}x — acceptable for weighted CE loss")

