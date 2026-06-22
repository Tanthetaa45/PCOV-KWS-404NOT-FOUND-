import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# ATTENTION & REGULARIZATION BLOCKS
# ─────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    CRITICAL FOR -5dB SNR: Dynamically recalibrates channel weights, 
    suppressing MUSAN/RIR noise and exciting human formants.
    """
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU6(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hardsigmoid(scale)
        return x * scale


# ─────────────────────────────────────────────
# DS-CNN BACKBONE WITH RESIDUALS
# ─────────────────────────────────────────────

class DSConvBlock(nn.Module):
    """
    Depthwise Separable Convolution Block + Residuals + SE Attention.
    Uses ReLU6 to bound activations to [0,6] — critical for NPU INT8 deployment.
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.1):
        super().__init__()
        self.use_res = (in_channels == out_channels) and (stride == 1 or stride == (1, 1))

        self.dw = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1,
            stride=stride, groups=in_channels, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.act_dw = nn.ReLU6(inplace=True)

        self.pw = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels)
        
        self.act_pw = nn.ReLU6(inplace=True)
        self.drop = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        identity = x
        
        out = self.act_dw(self.bn_dw(self.dw(x)))
        out = self.bn_pw(self.pw(out))
        out = self.se(out)

        if self.use_res:
            out = out + identity

        out = self.act_pw(out)
        out = self.drop(out)
        return out


# ─────────────────────────────────────────────
# HEADS
# ─────────────────────────────────────────────

class SVHead(nn.Module):
    """
    Speaker Verification Head — Early Exit at Block 2.
    Hidden dimension expanded to 512 to hit the 2.4M parameter target.
    """
    def __init__(self, in_dim=256, hidden_dim=512, embedding_dim=192, dropout_p=0.15):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=1)


class KWSHead(nn.Module):
    """
    Keyword Spotting Head — Deep Exit at Block 4.
    """
    def __init__(self, in_dim=512, num_classes=4, dropout_p=0.15):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.classifier(x)


# ─────────────────────────────────────────────
#  COMPLETE PCOV-KWS MODEL (2.4M PARAMETERS)
# ─────────────────────────────────────────────

class PCOVKWS(nn.Module):
    """
    PCOV-KWS-Edge: Asymmetric Dual-Head DS-CNN.
    Target: ~2.4M parameters (Perfectly matches PPT presentation).
    """
    def __init__(self, num_kws_classes=4, sv_embedding_dim=192, dropout_p=0.15):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True)
        )
        
        # Block 1: Shared Extractor
        self.block1 = nn.Sequential(
            DSConvBlock(64, 128, stride=1, dropout_p=dropout_p),
            DSConvBlock(128, 128, stride=1, dropout_p=dropout_p)
        )
        
        # Block 2: Shared & SV Exit (Retains 80-bin freq resolution)
        self.block2 = nn.Sequential(
            DSConvBlock(128, 256, stride=(1, 2), dropout_p=dropout_p),
            DSConvBlock(256, 256, stride=1, dropout_p=dropout_p),
            DSConvBlock(256, 256, stride=1, dropout_p=dropout_p)
        )

        self.sv_head = SVHead(in_dim=256, hidden_dim=512, embedding_dim=sv_embedding_dim, dropout_p=dropout_p)

        # Block 3: Deep KWS Path (Expanded to 4 layers for ~2.4M total params)
        self.block3 = nn.Sequential(
            DSConvBlock(256, 512, stride=(2, 2), dropout_p=dropout_p),
            DSConvBlock(512, 512, stride=1, dropout_p=dropout_p),
            DSConvBlock(512, 512, stride=1, dropout_p=dropout_p),
            DSConvBlock(512, 512, stride=1, dropout_p=dropout_p) # 🔥 Added 4th layer for max abstraction
        )
        
        # Block 4: Final KWS Abstraction
        self.block4 = nn.Sequential(
            DSConvBlock(512, 512, stride=(1, 2), dropout_p=dropout_p),
            DSConvBlock(512, 512, stride=1, dropout_p=dropout_p)
        )

        self.kws_head = KWSHead(in_dim=512, num_classes=num_kws_classes, dropout_p=dropout_p)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        
        shared_sv = self.block2(x)  
        sv_embedding = self.sv_head(shared_sv)

        x = self.block3(shared_sv)
        shared_kws = self.block4(x)
        kws_logits = self.kws_head(shared_kws)
        
        return sv_embedding, kws_logits

if __name__ == "__main__":
    model = PCOVKWS()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} (Target: < 3,000,000)")
    
    # Test forward pass with dummy frontend output
    dummy_input = torch.randn(8, 1, 80, 99) # [Batch, Channels, Mels, Time]
    sv_emb, kws_logits = model(dummy_input)
    
    print(f"SV Embedding Shape: {sv_emb.shape} (Expected: 8, 192)")
    print(f"KWS Logits Shape: {kws_logits.shape} (Expected: 8, 4)")

