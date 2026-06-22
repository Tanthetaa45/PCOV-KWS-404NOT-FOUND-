# Agentic Workflow — PCOV-KWS-Edge

## Overview

This document describes the agentic development process for PCOV-KWS-Edge,
including the AI-assisted debugging, synthetic data generation, and iterative
training pipeline that produced the final submission.

## Agentic Tools Used

| Tool | Purpose |
|---|---|
| Claude (Anthropic) | Architecture design, code review, training diagnostics |
| Coqui TTS | Synthetic wake-word generation (44,000 clips, 109 VCTK voices) |
| SpeechBrain ECAPA-TDNN | Teacher model for speaker verification knowledge distillation |
| Kaggle GPU (T4) | Training infrastructure (50 epochs, ~16 hours) |

## Agentic Workflows & Tool Chaining

Our agentic development setup relied heavily on **Claude** acting as the primary reasoning and planning orchestrator. 

- **Reasoning & Planning Pipelines:** Claude was used to generate an initial blueprint of the asymmetric DS-CNN architecture. From there, we established an iterative loop where failed runs (e.g., dead SV head gradients) were analyzed by feeding tracebacks and classification reports back to Claude's context. Claude reasoned about the interaction between PCGrad and AMP (double-unscale bug) to propose the targeted fixes.
- **Tool Use & Harness:** We used a local python testing harness to validate small architectural changes before pushing to Kaggle for full 50-epoch runs. We utilized Claude's coding assistant capabilities to write the `evaluate.py` test suite and standard metric trackers.
- **Memory & Context Handling:** To avoid context limits during long debugging sessions, we maintained a running `agents.md` scratchpad that recorded exact tensor shapes and gradient norms across iterations. This allowed the coding assistant to maintain global state awareness without re-reading the entire codebase.

## Development Timeline

### Phase 1 — Architecture & Dataset

Built the asymmetric dual-head DS-CNN architecture extending PCOV-KWS
(Pan & Huang, 2026). Generated 44,000 synthetic "Hey Galaxy" clips using
Coqui TTS across 109 VCTK speaker voices. Balanced the training set to
7.5:1 class ratio (improved from a naive 81:1 that caused training instability).

### Phase 2 — Training Iterations (4 rounds)

**Round 1 failed:** lr=3e-3 caused weight divergence at epoch 5. Wake Word
recall collapsed to 0%.

**Round 2 failed:** PCGrad + AMP interaction bug. Manual `inv_scale` inside
`pc_backward()` caused double-unscale — effective gradients were 65,536×
too small. Model appeared to train but SV head was "dead" (loss stuck at
0.71–0.75). CosineEmbeddingLoss has no angular margin, providing no
discriminative push between speakers.

**Round 3 failed:** Correct LR but SV head still dead. PCGrad permanently
biased KWS gradients away from SV direction since SV provided no useful
signal. Epoch 14–20 collapse confirmed this as the root cause.

**Round 4 succeeded — all 5 fixes combined:**
1. Fixed PCGrad AMP double-unscale bug
2. Replaced CosineEmbeddingLoss → SVEmbeddingKDLoss
3. SV head frozen epochs 0–14, unfrozen at epoch 15
4. lr=5e-4 (linear-scaled from batch128 baseline for batch64)
5. max_norm=1.0 (tighter clip for PCGrad's summed gradients)

Result: Clean convergence, no collapse, best model at epoch 47.

### Phase 3 — Live Demo Problem & Fix

After all offline KPIs passed (99.1% clean, 94.1% at -5dB), real microphone
testing revealed the model classified live "Hey Galaxy" as "Unknown" (confidence
0.07). Root causes:

1. **Synthetic-to-real domain gap:** 35,120 synthetic TTS clips vs ~279 real
   recordings (126:1). Model learned TTS spectral artifacts, not phonetics.
2. **Preprocessing mismatch:** Live audio was un-normalized. Verified via
   bit-identical tensor comparison — paths were different only in peak
   normalization step.

Fix: Surgical head-only fine-tune (407,044 trainable params / 17.23% of
total). Recorded 157 real clips, 25× augmentation → 3,692 real-derived
samples. MUSAN+RIR enabled in fine-tune train loader (previously trained
clean). Negative flooding: Unknown 2,500 + HardNeg 1,500. Dual checkpoint
gate: real-voice fire rate × KPI health multiplier.

Result: Real-voice confidence 0.07 → 0.94–0.998. All KPIs preserved.

## Key Lessons

1. Synthetic data trains models on spectral artifacts, not phonetic content.
   Real data must dominate the wake-word class even at small absolute counts.

2. PCGrad + AMP requires a single unscale pass after projection, not inside
   pc_backward(). The standard AMP pattern (scale → backward → unscale → clip
   → step) must be preserved even when manually handling gradients.

3. Staged SV head unfreezing is essential for joint KWS+SV training. An
   unstable SV head during early training permanently corrupts PCGrad's
   gradient surgery direction for KWS.

4. Evaluation must match training exactly. Every preprocessing step
   (peak normalization, crop strategy, resampling order) must be identical
   across train/val/evaluate/inference or evaluation metrics are meaningless
   for real deployment.

## Reproducibility

Full training pipeline runs on Kaggle (T4 GPU, free tier):
- See `src/notebook/01_PCOV_KWS_Base_Training.ipynb`
- See `src/notebook/02_PCOV_KWS_User_Finetune_Eval.ipynb`
- Reproducibility video: [YouTube](https://youtu.be/RrTlUff76JM)
