import time
import collections
import numpy as np
import torch
import sounddevice as sd

# ── IMPORT YOUR MODULAR STAGES ───────────────────────────────────────
from stage_zero import check_stage_zero
from frontend import PCENFrontend
from model import PCOVKWS

# ── PIPELINE CONFIGURATION CONSTANTS ─────────────────────────────────
SAMPLE_RATE   = 16000
CHUNK_SIZE    = 480       # 30ms streaming processing chunks
BUFFER_SIZE   = 16000     # Exact 1.0-second sliding array constraint
KWS_THRESHOLD = 0.85      # Wake word classification floor
SV_THRESHOLD  = 0.82      # Biometric speaker validation floor
TARGET_INDEX  = 0         # 🎯 WakeWord = index 0
WAKE_BIAS     = 0.4       # MUST match evaluate.py / check.py
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"📦 Loading optimized model weights onto device: [{DEVICE}]...")
checkpoint_path = "checkpoints/pcov_kws_finetuned.pth"
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

print("="*50)
print("🔍 Checkpoint Top-Level Keys:", list(checkpoint.keys()))
print("📊 Frontend Weights Key Present:", 'frontend_state_dict' in checkpoint)
print("="*50)

# Initialize neural architectures
model = PCOVKWS().to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint, strict=True)
model.eval()

frontend = PCENFrontend().to(DEVICE)
if 'frontend_state_dict' in checkpoint:
    frontend.load_state_dict(checkpoint['frontend_state_dict'])
frontend.eval()

# Warmup pass to eliminate lazy initialization latency spike
print("🔥 Running dummy model warmup pass...")
with torch.no_grad():
    _dummy_feat = frontend(torch.zeros(1, 1, 16000, device=DEVICE))
    _ = model(_dummy_feat)


def normalize_for_model(x):
    """Match training/eval preprocessing: /peak * 0.9"""
    return x / (np.max(np.abs(x)) + 1e-6) * 0.9


# ── BIOMETRIC SPEAKER PROFILE ENROLLMENT ─────────────────────────────
def run_live_enrollment():
    print("\n" + "="*60)
    print("👤 TEAM 404NOT-FOUND — VOICE EMBEDDING ENROLLMENT".center(60))
    print("="*60)
    print("Calibrating authorized user profile. Clear the room of background noise.")

    anchor_embeddings = []
    i = 0
    while i < 5:
        input(f"\nPress [ENTER], wait for GO, then say 'Hey Galaxy' (Sample {i+1}/5)...")
        print("🎙️ Recording in 1...")
        time.sleep(0.4)
        print("   GO! 🎙️ (1 second flat)")

        audio = sd.rec(int(1.0 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()

        raw_window = audio.flatten()

        rms = float(np.sqrt(np.mean(raw_window**2)))
        peak = float(np.max(np.abs(raw_window)))
        print(f"   [enroll debug] RMS={rms:.4f}  peak={peak:.3f}")

        # Stage 0 on RAW audio (thresholds tuned on raw)
        if not check_stage_zero(raw_window, sample_rate=SAMPLE_RATE, verbose_silence=False):
            print("⚠️ Sample rejected: Too silent or distorted. Please try again.")
            continue

        # Normalize ONLY for the model (match training/eval)
        window = normalize_for_model(raw_window)

        wave_tensor = torch.from_numpy(window).view(1, 1, -1).to(DEVICE)
        with torch.no_grad():
            feats = frontend(wave_tensor)
            embedding, _ = model(feats)          # SV embedding (already L2-normalized)
            anchor_embeddings.append(embedding)
        i += 1

    master_key = torch.mean(torch.stack(anchor_embeddings), dim=0)
    master_key = master_key / master_key.norm(p=2, dim=-1, keepdim=True)

    print("\n[enroll debug] pairwise self-similarity of your 5 samples:")
    embs = [e / e.norm(p=2, dim=-1, keepdim=True) for e in anchor_embeddings]
    sims = []
    for a in range(len(embs)):
        for b in range(a + 1, len(embs)):
            sims.append(torch.mm(embs[a], embs[b].t()).item())
    print(f"   self-sim min={min(sims):.2f}  mean={np.mean(sims):.2f}  max={max(sims):.2f}")
    print("   (mean should be HIGH, ~0.9+.)")
    
    # ── ADD HERE ──
    if np.mean(sims) < 0.80 or min(sims) < 0.60:
        print(f"⚠️ Enrollment weak (mean={np.mean(sims):.2f}, min={min(sims):.2f}). Re-enrolling...")
        return run_live_enrollment()   # restart enrollment from scratch
    print("✅ Strong enrollment anchor.")


    print("\n✅ Biometric vector identity calculated and secured in RAM!")
    return master_key


# ── LIVE AUDIO SLIDING WINDOW INFERENCE ENGINE ───────────────────────
def start_stream_inference(master_key):
    print("\n" + "="*60)
    print("🚀 EDGE INFERENCE CORE RUNNING — LISTENING".center(60))
    print("="*60)

    audio_buffer = collections.deque(maxlen=BUFFER_SIZE)
    score_history = collections.deque(maxlen=5)
    score_history.extend([0.0] * 5)
    biometric_sim = 0.0

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"\n⚠️ Hardware audio status alert: {status}", flush=True)
        audio_buffer.extend(indata[:, 0])

    mic_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                callback=mic_callback, blocksize=CHUNK_SIZE)

    print("\n⏳ Activating microphone stream and purging initial hardware noise...")
    with mic_stream:
        time.sleep(0.5)
        audio_buffer.clear()
        audio_buffer.extend(np.zeros(BUFFER_SIZE, dtype=np.float32))

        print("🎙️ Clean pipeline engaged! Listening...\n")

        try:
            while True:
                raw_window = np.array(audio_buffer, dtype=np.float32)

                if len(raw_window) != BUFFER_SIZE:
                    continue

                # Stage 0 on RAW audio
                if not check_stage_zero(raw_window, sample_rate=SAMPLE_RATE, verbose_silence=False):
                    print("\r🎙️ Listening... [Stage 0: Room Silent/Idle]                          ", end="", flush=True)
                    time.sleep(0.1)
                    continue

                start_latency_timer = time.perf_counter()

                # Normalize ONLY for the model (match training/eval)
                window = normalize_for_model(raw_window)
                window_tensor = torch.from_numpy(window).view(1, 1, -1).to(DEVICE)

                with torch.no_grad():
                    features = frontend(window_tensor)
                    speaker_vec, kws_logits = model(features)

                    kws_logits[:, TARGET_INDEX] += WAKE_BIAS          # ← match eval
                    probs = torch.softmax(kws_logits, dim=-1).squeeze(0).cpu().numpy()
                    raw_ww_prob = probs[TARGET_INDEX]

                    score_history.append(raw_ww_prob)
                    smoothed_kws_score = np.max(score_history)

                    speaker_vec = speaker_vec / speaker_vec.norm(p=2, dim=-1, keepdim=True)
                    biometric_sim = torch.mm(speaker_vec, master_key.t()).item()

                elapsed_ms = (time.perf_counter() - start_latency_timer) * 1000

                kws_pass = smoothed_kws_score >= KWS_THRESHOLD
                sv_pass  = biometric_sim >= SV_THRESHOLD

                print(
                    f"\rKWS={smoothed_kws_score:.2f}[{'Y' if kws_pass else 'n'}] "
                    f"SV={biometric_sim:+.2f}[{'Y' if sv_pass else 'n'}] "
                    f"argmax={int(np.argmax(probs))} "
                    f"p0={probs[0]:.2f} p1={probs[1]:.2f} p2={probs[2]:.2f} p3={probs[3]:.2f} "
                    f"{elapsed_ms:.0f}ms      ",
                    end="", flush=True
                )

                if smoothed_kws_score >= KWS_THRESHOLD:
                    if biometric_sim >= SV_THRESHOLD:
                        print(f"\n🟢 [ACCESS GRANTED] KWS: {smoothed_kws_score:.2f} | SV Sim: {biometric_sim*100:.1f}% | Latency: {elapsed_ms:.2f}ms", flush=True)
                    else:
                        print(f"\n🔴 [SPOOF REJECTED] KWS: {smoothed_kws_score:.2f} | SV Sim: {biometric_sim*100:.1f}% | Latency: {elapsed_ms:.2f}ms", flush=True)
                    score_history.extend([0.0] * 5)  # Debounce flush

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n🛑 Live streaming engine gracefully shut down.")


if __name__ == "__main__":
    authorized_profile = run_live_enrollment()
    start_stream_inference(authorized_profile)
