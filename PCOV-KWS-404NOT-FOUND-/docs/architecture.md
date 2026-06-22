# PCOV-KWS-Edge — System Architecture

## Overview

PCOV-KWS-Edge is a 2.36M parameter asymmetric dual-head DS-CNN for joint
Keyword Spotting (KWS) and Speaker Verification (SV) on Samsung Exynos NPU.
Extends PCOV-KWS (Pan & Huang, arXiv:2603.18023) with three architectural
innovations.

## Pipeline

```
Raw Audio (16kHz mono)
        │
        ▼
[Stage 0] Input Guard
  • NaN / Inf check
  • RMS silence gate (< 0.001)
  • Clipping detection (≥ 0.99)
  • Duration check (< 0.5s)
        │
        ▼
[Stage 1] PCEN Frontend
  • PreEmphasis (coef=0.97)
  • MelSpectrogram (80-bin, 400-pt FFT, 10ms hop)
  • Learned PCEN (inverse-sigmoid init, PCGrad-safe)
  • SpecAugment (training only)
        │
        ▼
[Stage 2] Shared DS-CNN Backbone

  Stem (1→64ch)
        │
        ▼
  Block1: DSConv 64→128 × 2
        │
        ▼
  Block2: DSConv 128→256 × 3 ──────────────→ [SV Head]
        │                                      AvgPool → Linear(256→512→192)
        ▼                                      L2-normalize → 192-dim embedding
  Block3: DSConv 256→512 × 4
        │
        ▼
  Block4: DSConv 512→512 × 2 ──────────────→ [KWS Head]
                                               AvgPool → Dropout → Linear(512→4)
                                               4-class: WW / Silence / Unknown / HardNeg
        │
        ▼
[Stage 3] Dual-Gate Decision
  Gate 1 (KWS): WakeWord prob ≥ 0.90 AND margin over 2nd class ≥ 0.40
  Gate 2 (SV):  cosine(embedding, enrolled_profile) ≥ 0.83
        │
        ▼
  🟢 ACCESS GRANTED / 🔴 REJECT
```

## Key Design Decisions

### Depthwise Separable Convolution (DS-CNN)
16× fewer FLOPs than standard convolution. Each DSConvBlock: depthwise
conv → BN → ReLU6 → pointwise conv → BN → SE attention → residual add → ReLU6 → Dropout2d.

### SE (Squeeze-and-Excitation) Attention
Dynamically recalibrates channel weights per input frame. Critical for
-5dB SNR robustness — suppresses noise channels, excites speech formant channels.

### Asymmetric Dual Exits
- SV exits at Block2 (80-bin resolution): captures phonetic/speaker features
- KWS exits at Block4 (downsampled): captures semantic/keyword features
- Prevents task interference: each head sees the feature resolution it needs

### PCEN (Per-Channel Energy Normalization)
Replaces log-mel. Learnable parameters (α, δ, r, s) initialized via
inverse-sigmoid/softplus for numerical stability. SNR-invariant by design —
handles -5dB to +30dB without explicit gain control.

### PCGrad Gradient Surgery
At each training step, SV and KWS gradients are projected onto each
other's normal plane before the optimizer step. Eliminates task conflict
without requiring separate optimizers. AMP-safe: projection operates on
scaled gradients, single unscale before clip_grad_norm_.

### ReLU6 Activations
Bounds activations to [0, 6]. Required for INT8 NPU quantization — prevents
large activations from overflowing the quantization range.

## Parameter Budget

| Component | Parameters |
|---|---|
| PCEN Frontend | 4 |
| Stem | 1,792 |
| Block1 | 25,344 |
| Block2 | 133,632 |
| SV Head (256→512→192) | 148,800 |
| Block3 | 1,083,392 |
| Block4 | 528,384 |
| KWS Head (512→4) | 2,052 |
| **Total** | **2,362,952** |

## Training Configuration

| Setting | Value |
|---|---|
| Optimizer | AdamW (lr=5e-4, weight_decay=1e-4) |
| Scheduler | OneCycleLR (max_lr=5e-4, pct_start=0.1) |
| Batch size | 64 |
| Epochs | 50 |
| Loss (KWS) | CrossEntropyLoss (inverse-freq class weights) |
| Loss (SV) | SVEmbeddingKDLoss (α=0.5) |
| SV freeze | Epochs 0–14 frozen, unfrozen at epoch 15 |
| Gradient clip | max_norm=1.0 |
| Mixed precision | torch.amp.autocast (CUDA) |
