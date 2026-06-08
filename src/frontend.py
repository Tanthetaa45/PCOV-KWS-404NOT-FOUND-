%%writefile frontend.py
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────
#  DSP PRE-PROCESSING
# Note: As per architecture, WebRTC VAD and RNNoise are executed 
# on the C++ DSP layer prior to this PyTorch pipeline to save power.
# ─────────────────────────────────────────────


class PreEmphasis(nn.Module):
    """
    Boosts high frequencies to balance the spectrum of the human voice.
    Matches the 'Stage 1: Pre-emphasis' claim in the PPT.
    """
    def __init__(self, coef=0.97):
        super().__init__()
        self.coef = coef
        # FIR filter weight: y[t] = x[t] - coef * x[t-1]
        self.register_buffer('weight', torch.tensor([-self.coef, 1.0]).view(1, 1, 2))

    def forward(self, x):
        # x: (Batch, 1, Time)
        x_padded = F.pad(x, (1, 0), 'reflect')
        return F.conv1d(x_padded, self.weight)
    
    
# ─────────────────────────────────────────────
#  PCEN FEATURE EXTRACTION
# ─────────────────────────────────────────────

class PCENTransform(nn.Module):
    """
    Per-Channel Energy Normalization (Wang et al., ICASSP 2017).
    [UPGRADED] Parameters are constrained via Sigmoid/Softplus to prevent NaN crashes.
    """
    def __init__(self, gain=0.98, bias=2.0, power=0.5,
                 time_constant=0.025, eps=1e-6):
        super().__init__()
        self.eps = eps

        # ---------------------------------------------------------
        # INITIALIZATION USING INVERSE MATH
        # We use Inverse-Sigmoid (Logit) and Inverse-Softplus so that 
        # the parameters start EXACTLY at the default values.
        # ---------------------------------------------------------
        
        # Sigmoid forces alpha, r, and s to stay safely between (0.0, 1.0)
        # logit(x) = ln(x / (1 - x))
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(gain / (1.0 - gain))))
        self.raw_r     = nn.Parameter(torch.tensor(math.log(power / (1.0 - power))))
        self.raw_s     = nn.Parameter(torch.tensor(math.log(time_constant / (1.0 - time_constant))))
        
        # Softplus forces delta to stay strictly positive (> 0.0)
        # inv_softplus(x) = ln(exp(x) - 1)
        self.raw_delta = nn.Parameter(torch.tensor(math.log(math.exp(bias) - 1.0)))

    def forward(self, x):
        # 1. Apply safety activations to the raw learnable parameters
        alpha = torch.sigmoid(self.raw_alpha)
        r     = torch.sigmoid(self.raw_r)
        s     = torch.sigmoid(self.raw_s)
        delta = F.softplus(self.raw_delta)

        # 2. Standard PCEN computation using the safe variables
        # [PCGRAD FIX] list+stack instead of in-place writes
        # In-place m[...,t]= breaks retain_graph=True in PCGrad second backward.
        # [ONNX WARNING] Dynamic loop needs JIT script or 1D conv for export.py
        frames = [x[..., 0]]
        for t in range(1, x.shape[-1]):
            frames.append((1.0 - s) * frames[-1] + s * x[..., t])
        m = torch.stack(frames, dim=-1)
            
        return ((x / (self.eps + m) ** alpha) + delta) ** r - delta ** r


class PCENFrontend(nn.Module):
    """
    Stages 1+2: Raw waveform → 80-bin PCEN Mel-Spectrogram.
    Includes SpecAugment (Time/Frequency Masking) active ONLY during training.
    """
    def __init__(self, sample_rate=16000, n_mels=80):
        super().__init__()
        self.pre_emphasis = PreEmphasis(coef=0.97)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,        
            hop_length=160,   
            n_mels=n_mels,
            f_min=20,
            f_max=8000,
            power=2.0         # Explicit declaration per hackathon best-practices
        )
        self.pcen = PCENTransform()
        
        # --- SPECAUGMENT ---
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)

    def forward(self, waveform):
        x = self.pre_emphasis(waveform)  # FIR boost consonants before mel
        mel = self.mel(x.squeeze(dim=1))  # squeeze(dim=1) — safe for batch=1   
        pcen_out = self.pcen(mel)             
        
        if self.training:
            pcen_out = self.freq_mask(pcen_out)
            pcen_out = self.time_mask(pcen_out)

        return pcen_out.unsqueeze(1)          
