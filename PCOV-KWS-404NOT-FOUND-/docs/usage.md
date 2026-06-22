# Usage Guide

## 1. Run Official KPI Evaluation

Measures all 8 PS4 KPIs against the validation set:

```bash
python evaluate.py \
    --checkpoint src/checkpoints/pcov_kws_best.pth \
    --val_manifest data/manifests/val_manifest.json \
    --musan_dir data/raw/musan \
    --output_dir results/
```

Outputs:
- `results/kpi_evaluation.csv` — raw numbers
- `results/kpi_evaluation.md` — copy-ready for reports

## 2. Live Microphone Demo

Runs the full streaming pipeline — ring buffer + sliding window + dual gate:

```bash
python mic_demo.py \
    --speaker abhishek \
    --checkpoint src/checkpoints/pcov_kws_finetuned.pth \
    --profiles src/checkpoints/enrolled_profiles.pt
```

**Three demo scenarios:**

**Scenario 1 — Authorized speaker**

Say "Hey Galaxy" → KWS fires → SV matches enrolled profile

🟢 ACCESS GRANTED — Abhishek verified (SV score: 0.94, latency: 14ms)

**Scenario 2 — Live enrollment (no retraining)**

```bash
python mic_demo.py --enroll --speaker newuser --n_clips 3
```

Speak "Hey Galaxy" 3 times → profile saved → immediately active

**Scenario 3 — Unauthorized speaker**

Different person speaks → KWS fires → SV gate rejects

🔴 SPOOF REJECTED — speaker not authorized (SV score: 0.61 < threshold 0.83)

## 3. Fine-Tune for Your Voice

Personalize the KWS head to your own voice (head-only, ~45 min on GPU):

```bash
# Step 1: Record your clips (place in data/raw/target_kws/{your_name}/)
# Step 2: Augment
python src/data/augment_real.py

# Step 3: Build fine-tune manifest
python src/data/build_finetune_manifest.py

# Step 4: Fine-tune KWS head only
python tune_kws.py
```

Output: `src/checkpoints/pcov_kws_finetuned.pth`

## 4. Train From Scratch

Full 50-epoch training (requires ~16 hours on T4 GPU):

```bash
python train.py
```

See `src/notebook/01_PCOV_KWS_Base_Training.ipynb` for the Kaggle version
with dataset download cells and session-resume logic.

## 5. Stage 0 Input Guard (standalone)

```python
from stage_zero import check_stage_zero
import numpy as np

audio = np.zeros(16000)  # 1s of silence
result = check_stage_zero(audio)
# False — silence detected, inference skipped
```

## Configuration

Key thresholds (edit in respective files):

| Parameter | Default | File | Effect |
|---|---|---|---|
| `KWS_THRESHOLD` | 0.90 | `mic_demo.py` | Min WakeWord confidence to trigger |
| `MARGIN_THRESHOLD` | 0.40 | `mic_demo.py` | Min gap over 2nd-best class |
| `SV_THRESHOLD` | 0.83 | `mic_demo.py` | Min cosine similarity for speaker match |
| `SILENCE_RMS_THRESHOLD` | 0.001 | `stage_zero.py` | Lower if soft speech rejected |
| `HOP_MS` | 10 | `mic_demo.py` | Sliding window hop (ms) — lower = finer detection |
