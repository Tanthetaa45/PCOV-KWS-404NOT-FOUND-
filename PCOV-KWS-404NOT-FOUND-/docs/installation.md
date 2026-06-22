# Installation Guide

## Requirements

- Python 3.10+
- CUDA-capable GPU (for training) — inference runs on CPU
- ~5GB disk space for model weights and dependencies

## Quick Install

```bash
git clone https://github.com/404NOT-FOUND/PCOV-KWS-404NOT-FOUND
cd PCOV-KWS-404NOT-FOUND
pip install -r requirements.txt
```

## Model Weights

Model weights are included in the repository under `src/checkpoints/`.
Verify they are present after cloning:

```
src/checkpoints/
├── pcov_kws_best.pth        # Main model (all offline KPIs)
└── pcov_kws_finetuned.pth   # Fine-tuned (real-voice microphone detection)
```

## Verify Installation

```bash
python evaluate.py --checkpoint src/checkpoints/pcov_kws_best.pth
```

Expected output:

```
KPI 1  TAR Clean          99.10%   ✅ PASS
KPI 2  TAR Noisy (-5dB)   94.10%   ✅ PASS
KPI 3  FA Rate             0.019/hr ✅ PASS
KPI 6  Parameters      2,362,952   ✅ PASS
KPI 7  Latency            15.72ms  ✅ PASS
```

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| torch | 2.2.2 | Core deep learning |
| torchaudio | 2.2.2 | Audio loading, resampling |
| sounddevice | 0.4.6 | Live microphone input |
| numpy | 1.26.4 | Numerical operations |
| soundfile | 0.13.1 | WAV file I/O |
| librosa | 0.10.1 | Audio augmentation |
| scikit-learn | 1.4.2 | Metrics (EER, classification report) |
| speechbrain | 0.5.16 | ECAPA-TDNN teacher embeddings |
| tqdm | 4.66.2 | Progress bars |
| tabulate | 0.9.0 | Markdown table export |

## Platform Notes

- **Linux/macOS:** All features supported
- **Windows:** `sounddevice` mic input requires PortAudio — install via
  `pip install sounddevice` (PortAudio bundled on most systems)
- **Kaggle/Colab:** Use notebook versions in `src/notebook/`
- **CPU-only inference:** Remove `torch.cuda.is_available()` check and
  set `device = "cpu"` — latency increases to ~80–120ms (still under 200ms target)
