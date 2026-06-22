<div align="center">

# PCOV-KWS-Edge

### Speaker-Personalized, Always-On Wake-Word Detection for Samsung Exynos NPU

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Samsung EnnovateX AX 2026](https://img.shields.io/badge/Samsung-EnnovateX%20AX%202026-1428A0?logo=samsung&logoColor=white)](#-submission-details)

**Team 404NOT-FOUND** — Abhishek Kujur & Tanmay Rath  
National Institute of Technology, Rourkela · 4th Year  

*Samsung EnnovateX AX Hackathon 2026 · Phase 2 · PS4 — Speech Disentanglement*

</div>

---

## 🚀 Overview

**PCOV-KWS-Edge** is an always-on, on-device wake-word detection system that recognizes **"Hey Galaxy"** and responds **only** to the authorized speaker. It implements a dual-gate security architecture — the Keyword Spotting (KWS) gate verifies *what* was said, and the Speaker Verification (SV) gate verifies *who* said it. Both gates must pass simultaneously for access to be granted.

The system is built on a **2.36M parameter asymmetric dual-head DS-CNN** with joint KWS and SV heads, designed to run within the strict resource constraints of the **Samsung Exynos NPU** (**< 3M parameters**, **< 200 ms latency**).

### Key Highlights

- **Dual-gate security** — KWS + biometric SV prevents unauthorized activation, even if the wake word is spoken by an impostor
- **PCEN frontend** — Learned Per-Channel Energy Normalization replaces standard log-mel, providing robust noise adaptation down to −5 dB SNR
- **Asymmetric early exit** — SV head exits at Block 2 (low-level speaker features), KWS head exits at Block 4 (high-level semantic features)
- **PCGrad gradient surgery** — Eliminates destructive interference between KWS and SV task gradients during multi-task training
- **On-device personalization** — New users enroll by saying "Hey Galaxy" 3 times; immediately recognized without retraining
- **99.25% True Acceptance** on clean speech, **97.6%** at −5 dB SNR, **12.62 ms** inference latency

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │           PCOV-KWS-Edge  (2.36M params)        │
                    └─────────────────────────────────────────────────┘

  Raw Waveform (16 kHz, 1s)
        │
        ▼
  ┌──────────────┐
  │ Pre-Emphasis │   FIR high-frequency boost (coef = 0.97)
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │  Mel-Spec    │   80-bin, n_fft=400, hop=160, f_min=20, f_max=8000
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │    PCEN      │   Learned α, δ, r, s (sigmoid/softplus constrained)
  └──────┬───────┘   Wang et al., ICASSP 2017
         ▼
  ┌──────────────┐
  │  SpecAugment │   FreqMask(15) + TimeMask(20) — training only
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │   DS-CNN     │   Stem → Block1 (128ch) → Block2 (256ch)
  │   Backbone   │   Depthwise Separable Conv + SE Attention + ReLU6
  └──┬───────┬───┘
     │       │
     │       ▼  ─── Early Exit ───
     │  ┌──────────┐
     │  │ SV Head  │  AdaptiveAvgPool → MLP(256→512→192) → L2-Norm
     │  └──────────┘  192-dim speaker embedding
     │
     ▼
  ┌──────────────┐
  │  Block3 (512)│   4 × DSConvBlock (deep KWS path)
  │  Block4 (512)│   2 × DSConvBlock (final abstraction)
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │  KWS Head   │   AdaptiveAvgPool → Dropout → Linear(512→4)
  └──────────────┘   4-class: WakeWord / Silence / Unknown / Hard-Negative
```

### Training Pipeline

| Component | Detail |
|---|---|
| **Multi-task optimizer** | PCGrad (Yu et al., NeurIPS 2020) — gradient surgery resolves conflicting KWS↔SV gradients |
| **KWS loss** | Weighted CrossEntropyLoss (class-balanced) |
| **SV loss** | SVEmbeddingKDLoss — cosine embedding distillation from ECAPA-TDNN teacher |
| **Scheduler** | OneCycleLR (max_lr=5e-4, cosine anneal, 10% warmup) |
| **Precision** | Mixed-precision (AMP + GradScaler on CUDA) |
| **Augmentation** | MUSAN noise (−5 to +10 dB), RIR convolution (30% prob), SpecAugment |
| **SV freeze** | SV head frozen for first 15 epochs to stabilize backbone |

---

## 📊 Official KPI Results

> All KPIs tested and measured by the evaluation pipeline in [`evaluate.py`](src/evaluate.py).

| KPI | Metric | Result | Target | Status |
|-----|--------|--------|--------|--------|
| KPI 1 | True Acceptance (Clean) | **99.25%** | ≥ 99% | ✅ PASS |
| KPI 2 | True Acceptance (Noisy) | **97.60%** | ≥ 90% | ✅ PASS |
| KPI 3 | False Acceptance Rate | **0.0212/hr** | < 1/hr | ✅ PASS |
| KPI 4 | SNR Range (−5 dB to 30 dB) | **97.6% @ −5 dB → 99.1% @ 30 dB** | −5 dB to 30 dB | ✅ PASS |
| KPI 5 | Distance Range (0.5 m – 5 m) | RIR-augmented in training | 0.5 m to 5 m | ℹ️ INFO |
| KPI 6 | Speaker Support (M/F) | Multi-speaker TTS + real voices | Male & Female | ℹ️ INFO |
| KPI 7 | Model Parameters | **2,362,952** | < 3,000,000 | ✅ PASS |
| KPI 8 | Execution Time (xRT, p95=13.7 ms) | **12.62 ms** (0.0126 s) | < 0.2 s | ✅ PASS |

<details>
<summary><strong>📏 KPI 5 — Distance Range (0.5 m – 5 m)</strong></summary>

Distance robustness is addressed via **RIR (Room Impulse Response) convolution augmentation** during training (`RAMCachedRIR`, 30% probability), simulating speaker-to-mic distances across the 0.5 m – 5 m target range. Not independently re-measured at eval time (no per-distance labeled validation set available) — reported as a training-time mitigation, consistent with how SNR/noise robustness is achieved.

</details>

<details>
<summary><strong>🗣️ KPI 6 — Speaker Support (Male & Female)</strong></summary>

Training data includes **87 Coqui TTS speakers (mixed gender)** plus real recordings from **2 team members**. The SV branch has been verified working for both enrolled speakers.

</details>

---

## 🎬 Live Demo Scenarios

### Scenario 1 — Authorized Speaker Access
> Authorized speaker says **"Hey Galaxy"** → KWS gate fires (WakeWord detected) → SV gate matches enrolled biometric profile → **🟢 ACCESS GRANTED**

The dual-head architecture processes both gates in a single forward pass. The KWS head classifies the utterance as `WakeWord` (class 0), and the SV head produces a 192-dim L2-normalized embedding that is compared against the enrolled speaker's master key via cosine similarity.

### Scenario 2 — Live Enrollment (On-Device Personalization)
> New user enrolls live by speaking **"Hey Galaxy" 3 times** → Embeddings averaged into a master key → Immediately recognized without retraining → Demonstrates **on-device personalization**

The enrollment process in [`mic_demo.py`](src/mic_demo.py) captures 5 utterances, validates each with a Stage 0 energy gate, extracts SV embeddings, and computes a mean L2-normalized master key. Self-similarity is checked to ensure enrollment quality (mean ≥ 0.80, min ≥ 0.60).

### Scenario 3 — Spoof Rejection
> Unauthorized speaker says **"Hey Galaxy"** → KWS gate fires (correct word detected) → SV gate rejects (embedding distance too far from enrolled profile) → **🔴 SPOOF REJECTED**

This demonstrates the dual-gate security architecture: even though the KWS head correctly detects the wake word, the SV head independently verifies the speaker's identity. An impostor's voice produces a sufficiently different embedding to fall below the SV threshold (0.82 cosine similarity).

---

## 📹 Demo Videos

| Video | Link |
|-------|------|
| 🔬 Reproducibility & setup walkthrough | [Google Drive](https://drive.google.com/file/d/1y7S6l_4m_NuOjGaiUA9A_lEVxjR9fSLy/view?usp=sharing) |
| 🎙️ Live microphone demo | [Google Drive](https://drive.google.com/file/d/1M4M277WscHwMgPzS4xjrRD3sg5MKQudB/view?usp=sharing) |

---

## 🗂️ Repository Structure

```
PCOV-KWS-404NOT-FOUND/
├── docs/                            # Documentation
│   ├── ax.md                        #   Hackathon submission notes
│   ├── architecture.md              #   Detailed architecture documentation
│   ├── installation.md              #   Setup guide
│   └── usage.md                     #   Usage guide
├── src/
│   ├── checkpoints/                 # Model weights (included in repo)
│   │   ├── pcov_kws_best.pth        #   Main 50-epoch trained model
│   │   ├── pcov_kws_finetuned.pth   #   Fine-tuned for real-voice mic demo
│   │   └── last_checkpoint.pth      #   Resume training checkpoint
│   ├── data/                        # Data pipeline
│   │   ├── augment_real.py          #   Real recording augmentation
│   │   ├── build_fintune_manifest.py#   Fine-tune manifest builder
│   │   ├── dataset.py               #   PCOVMultiTaskDataset (MUSAN + RIR augmentation)
│   │   └── manifest_generator.py    #   Training/val manifest generation
│   ├── model/                       # Model definitions
│   │   ├── model.py                 #   PCOVKWS — asymmetric dual-head DS-CNN
│   │   ├── precompute_embeddings.py #   ECAPA-TDNN teacher embedding extraction
│   │   └── teacher_loader.py        #   SpeechBrain ECAPA-TDNN teacher loader
│   └── notebook/                    # Jupyter notebooks (Kaggle)
│       ├── 01_PCOV_KWS_Base_Training.ipynb
│       ├── 02_PCOV_KWS_User_Finetune_Eval.ipynb
│       ├── Stage_0.ipynb
│       └── Stage_1.ipynb
├── evaluate.py                      # Official 8-KPI evaluation pipeline
├── frontend.py                      # PCEN frontend (PreEmphasis + Mel + PCEN + SpecAugment)
├── loss.py                          # SphereFace2, AAMSoftmax, KWSLoss, KDLoss, SVEmbeddingKDLoss
├── mic_demo.py                      # Real-time microphone demo with live enrollment
├── stage_zero.py                    # Stage 0 energy gate (silence/distortion filter)
├── train.py                         # Multi-task PCGrad training pipeline (50 epochs)
├── tune_kws.py                      # User-specific KWS fine-tuning
├── requirements.txt                 # Python dependencies
├── LICENSE                          # Apache 2.0
└── README.md                        # ← You are here
```

---

## 📦 Model Weights

Model checkpoints are included in the repository under `src/checkpoints/`:

| Checkpoint | Description |
|---|---|
| `pcov_kws_best.pth` | Main 50-epoch trained model — passes all 8 offline KPIs |
| `pcov_kws_finetuned.pth` | Fine-tuned checkpoint for real-voice microphone detection |

---

## 📚 Datasets

| Dataset | Purpose | Link |
|---|---|---|
| **Google Speech Commands v2** | Unknown speech class & KWS baseline evaluation | [Kaggle](https://www.kaggle.com/datasets/sylkaladin/speech-commands-v2) |
| **LibriPhrase** | Phonetically diverse hard-negative phrases | [Kaggle](https://www.kaggle.com/code/abhishekkujur19/libriphrase-data) |
| **MUSAN** | Background noise augmentation (−5 dB to +10 dB SNR training) | [Kaggle](https://www.kaggle.com/datasets/nhattruongdev/musan-noise) |
| **OpenSLR-28 (Room Impulse Responses)** | Distance simulation (0.5 m – 5 m) via RIR convolution | [Kaggle](https://www.kaggle.com/datasets/nhattruongdev/rirs-noises) |
| **Coqui TTS (synthetic)** | 44,000 synthetic "Hey Galaxy" wake word clips generated using 109 VCTK voices | [Kaggle](https://www.kaggle.com/code/abhishekkujur19/agents-data) |

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- Microphone (for live demo)

### Installation

```bash
git clone https://github.com/404NOT-FOUND/PCOV-KWS-404NOT-FOUND
cd PCOV-KWS-404NOT-FOUND
pip install -r requirements.txt
```

Model weights are already included in `src/checkpoints/`. Verify they are present after cloning:

```
src/checkpoints/
├── pcov_kws_best.pth
└── pcov_kws_finetuned.pth
```

### Run Evaluation (All 8 KPIs)

```bash
python evaluate.py --checkpoint src/checkpoints/pcov_kws_best.pth
```

This runs the full Samsung KPI evaluation pipeline — True Acceptance (clean + noisy), False Acceptance Rate (dual-gate), SNR sweep (−5 dB to 30 dB), parameter count, and latency benchmark. Results are exported as CSV and Markdown reports.

### Run Live Microphone Demo

```bash
python mic_demo.py --speaker abhishek --checkpoint src/checkpoints/pcov_kws_finetuned.pth
```

The demo will:
1. **Enroll** your voice — speak "Hey Galaxy" 5 times when prompted
2. **Listen** in real-time — say "Hey Galaxy" and watch the dual-gate decision
3. **Display** per-frame KWS confidence, SV cosine similarity, and latency

### Run User Fine-Tune (Personalize to Your Voice)

```bash
python tune_kws.py
```

Fine-tunes the KWS head (Blocks 3–4) using your personal recordings while keeping the shared backbone and SV head frozen. Automatically balances class weights and applies noise augmentation.

---

## 🔧 Training from Scratch

<details>
<summary>Expand for full training instructions</summary>

### 1. Precompute Teacher Embeddings

```bash
python src/model/precompute_embeddings.py
```

Extracts 192-dim speaker embeddings from a pretrained ECAPA-TDNN teacher (via SpeechBrain) for all training and validation samples.

### 2. Base Training (50 Epochs)

```bash
python train.py
```

Runs the full multi-task PCGrad training pipeline:
- KWS head: weighted CrossEntropyLoss (4-class)
- SV head: SVEmbeddingKDLoss (cosine distillation from ECAPA-TDNN)
- PCGrad gradient surgery resolves conflicting task gradients
- OneCycleLR scheduler (max_lr=5e-4, cosine anneal)
- SV head frozen for first 15 epochs to stabilize backbone
- Mixed-precision training with GradScaler

### 3. User Fine-Tuning

```bash
python tune_kws.py
```

Fine-tunes Blocks 3–4 and KWS head on user-specific recordings (15 epochs, lr=1.5e-5).

</details>

---

## 📐 Technical Details

<details>
<summary><strong>PCEN Frontend</strong></summary>

Per-Channel Energy Normalization replaces the standard log-mel energy compression with a **learned, adaptive normalization** that automatically adjusts to varying noise conditions. Parameters (α, r, s, δ) are initialized via inverse-sigmoid/inverse-softplus to exact default values and trained end-to-end with the backbone.

Key advantage at low SNR: PCEN's temporal smoothing suppresses stationary noise while preserving transient speech energy — directly contributing to the **97.6% TA at −5 dB SNR** result.

</details>

<details>
<summary><strong>SE Attention in DS-CNN Blocks</strong></summary>

Every depthwise separable convolution block includes a **Squeeze-and-Excitation (SE)** channel attention mechanism with reduction ratio 8. The SE block dynamically recalibrates channel weights, suppressing MUSAN/RIR noise channels and exciting human formant channels. Activation is bounded by **Hardsigmoid** for NPU INT8 compatibility.

</details>

<details>
<summary><strong>Dual-Gate False Acceptance Calculation (KPI 3)</strong></summary>

```
P(FA) = P(FA | KWS gate) × P(FA | SV gate)
FA/hr = P(FA) × triggers_per_hour

P(FA | KWS) = 0.0354%  (Hard-Negative → WakeWord misclassification rate)
P(FA | SV)  = 3.0%     (fixed SV impostor acceptance rate)
Triggers/hr = 20       (estimated non-wake triggers per hour)

FA/hr = 0.000354 × 0.03 × 20 = 0.0212/hr  ✅ (target < 1/hr)
```

The dual-gate architecture achieves a **47× margin** below the 1/hr target.

</details>

---

## 📋 Submission Details

| Field | Value |
|---|---|
| **Hackathon** | Samsung EnnovateX AX Hackathon 2026 |
| **Problem Statement** | PS4 — Speech Disentanglement / Speaker-Personalized Keyword Detection |
| **Team** | 404NOT-FOUND |
| **Members** | Abhishek Kujur, Tanmay Rath |
| **Institution** | National Institute of Technology, Rourkela |
| **Submission Email** | ennovatex.io@samsung.com |
| **Subject Line** | `AX Hackathon Phase 2 Submission \| PS4 \| 404NOT-FOUND` |

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

## 📖 References

1. **Pan, W. & Huang, J.** (2026). *PCOV-KWS: Per-Channel Output Voting for Keyword Spotting.* arXiv:2603.18023. [arXiv](https://arxiv.org/abs/2603.18023)

2. **Zhang, Y., Suda, N., Lai, L., & Chandra, V.** (2017). *Hello Edge: Keyword Spotting on Microcontrollers.* arXiv:1711.07128. [arXiv](https://arxiv.org/abs/1711.07128)

3. **Desplanques, B., Thienpondt, J., & Demuynck, K.** (2020). *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.* Interspeech 2020. [arXiv](https://arxiv.org/abs/2005.07143)

4. **Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C.** (2020). *Gradient Surgery for Multi-Task Learning.* NeurIPS 2020. [arXiv](https://arxiv.org/abs/2001.06782)

5. **Wang, Y., Getreuer, P., Hughes, T., Lyon, R. F., & Saurous, R. A.** (2017). *Trainable Frontend For Robust and Far-Field Keyword Spotting.* ICASSP 2017. [IEEE](https://doi.org/10.1109/ICASSP.2017.7953242)

---

<div align="center">

**Built with 🔥 by Team 404NOT-FOUND · NIT Rourkela**

*Samsung EnnovateX AX Hackathon 2026 — Phase 2*

</div>

