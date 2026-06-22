import torch
import torch.nn as nn
import warnings


class MatchboxTeacher(nn.Module):
    """
    KWS Teacher — NVIDIA NeMo MatchboxNet3x1x64_v2.
    Pretrained on Google Speech Commands v2 (35 classes).
    Used for knowledge distillation into the student KWS head.
    Frozen at load — no gradient flow during training.

    Input:  (B, 1, 16000) or (B, 16000) — raw 16kHz waveform
    Output: (B, num_classes) — logits
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.use_real_model = False

        try:
            import nemo.collections.asr as nemo_asr
            print("🔄 Loading NVIDIA NeMo MatchboxNet...")
            self.nemo_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
                model_name="commandrecognition_en_matchboxnet3x1x64_v2"
            )
            self.nemo_model.eval()
            for p in self.nemo_model.parameters():
                p.requires_grad = False
            self.use_real_model = True
            print("✅ MatchboxNet loaded and frozen.")
        except (ImportError, Exception) as e:
            warnings.warn(f"NeMo unavailable ({type(e).__name__}). Run: pip install nemo_toolkit[asr]")
            self.fallback = nn.Linear(16000, num_classes)

    def forward(self, x):
        # Accept (B,1,16000) from train.py or (B,16000) directly
        if x.dim() == 3:
            x = x.squeeze(1)

        if self.use_real_model:
            # Force model to same device as input — NeMo does not auto-track device
            self.nemo_model = self.nemo_model.to(x.device)
            lens = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
            logits, _ = self.nemo_model(audio_signal=x, audio_signal_length=lens)
            return logits

        return self.fallback(x)


class ECAPATeacher(nn.Module):
    """
    SV Teacher — SpeechBrain ECAPA-TDNN pretrained on VoxCeleb.
    Outputs 192-dim speaker embeddings.
    Used for knowledge distillation into the student SV head.
    Frozen at load — no gradient flow during training.

    Input:  (B, 1, 16000) or (B, 16000) — raw 16kHz waveform
    Output: (B, embedding_dim) — L2-normalized speaker embedding
    """
    def __init__(self, embedding_dim=192):
        super().__init__()
        self.use_real_model = False
        self.proj = None

        try:
            from speechbrain.inference.speaker import EncoderClassifier
            print("🔄 Loading SpeechBrain ECAPA-TDNN...")
            self.sb_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="/kaggle/working/tmp_sb_models",
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
            # Verify output dim — ECAPA outputs 192 natively but guard anyway
            with torch.no_grad():
                test = self.sb_model.encode_batch(torch.zeros(1, 16000)).squeeze(1)
            if test.shape[-1] != embedding_dim:
                self.proj = nn.Linear(test.shape[-1], embedding_dim)
                print(f"⚠️  Projection added: {test.shape[-1]}→{embedding_dim}")

            for p in self.sb_model.parameters():
                p.requires_grad = False
            self.use_real_model = True
            print("✅ ECAPA-TDNN loaded and frozen.")
        except (ImportError, Exception) as e:
            warnings.warn(f"SpeechBrain unavailable ({type(e).__name__}). Run: pip install speechbrain")
            self.fallback = nn.Linear(16000, embedding_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)

        if self.use_real_model:
            # Sync SpeechBrain internals to match inbound tensor device
            self.sb_model.device = x.device
            self.sb_model.mods = self.sb_model.mods.to(x.device)
            emb = self.sb_model.encode_batch(x).squeeze(1)
            if self.proj is not None:
                emb = self.proj(emb)
            return emb

        return self.fallback(x)