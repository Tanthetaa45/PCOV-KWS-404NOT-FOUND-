"""
stage_zero.py — Stage 0: Input Validation Guard
Team: 404NOT-FOUND | NIT Rourkela
Samsung EnnovateX AX Hackathon 2026 — PS4

"The AI never sees bad audio"

Validates raw microphone array buffers before executing any DSP or
neural network inference stages, to preserve edge battery life and
device stability. Keeps the heavy KWS/SV model asleep until valid
audio enters the frame.

Threshold notes:
  - Training-time silence guard (dataset.py) uses RMS < 1e-8 — appropriate
    for studio/synthetic audio with very quiet but phonetically valid fades.
  - This live-stream guard uses RMS < 1e-3 (0.001) by default — appropriate
    for real microphone ambient noise floor (fans, room echo, distant talk),
    keeping the model asleep during genuine silence to save battery.
  - If soft/quiet speech is being incorrectly rejected during live testing,
    lower SILENCE_RMS_THRESHOLD to 1e-4 or 5e-5.
"""

import numpy as np

SILENCE_RMS_THRESHOLD = 0.001    # tune down to 1e-4 / 5e-5 if soft speech is rejected
CLIP_THRESHOLD         = 0.99
MIN_DURATION_SECONDS   = 0.5


def check_stage_zero(audio_window, sample_rate=16000,
                      silence_threshold=SILENCE_RMS_THRESHOLD,
                      verbose_silence=False):
    """
    STAGE 0: INPUT VALIDATION GUARD

    Runs 4 cheap checks on a raw audio window before any DSP/neural
    network stage is allowed to execute. Returns True only if all
    gates pass cleanly.

    Args:
        audio_window: 1D numpy array of raw audio samples (float, [-1, 1] range)
        sample_rate: sample rate of audio_window, in Hz
        silence_threshold: RMS energy below which audio is treated as silence
        verbose_silence: if True, prints a log line on silence rejection too
                        (default False, to avoid flooding the console while
                        idle/listening with nobody speaking)

    Returns:
        bool — True if audio_window passes all 4 gates, False otherwise
    """

    # ── 1. CORRUPTED / NaN SAMPLES CHECK ──
    # Catches hardware glitches or streaming buffer overflows
    if np.isnan(audio_window).any() or np.isinf(audio_window).any():
        print("⚠️ [Stage 0] REJECTED: Corrupted buffer or NaN sample detected. "
              "| Log: ERR_CORRUPT")
        return False

    # ── 2. TOO SHORT CHECK ──
    # Insufficient frames for the 25ms STFT windows / unprimed startup buffer
    duration_seconds = len(audio_window) / sample_rate
    if duration_seconds < MIN_DURATION_SECONDS:
        print(f"⚠️ [Stage 0] REJECTED: Audio window too short "
              f"({duration_seconds:.2f}s < {MIN_DURATION_SECONDS}s). | Log: ERR_SHORT")
        return False

    # ── 3. COMPLETE SILENCE CHECK ──
    # RMS energy threshold prevents running inference on dead background static.
    # Tuned for live mic ambient noise floor, not studio/synthetic training audio.
    rms = np.sqrt(np.mean(audio_window ** 2))
    if rms < silence_threshold:
        if verbose_silence:
            print(f"ℹ️ [Stage 0] SILENCE: rms={rms:.6f} < {silence_threshold} "
                  f"— skipping inference path. | Log: ERR_SILENCE")
        # Fails silently by default to prevent console log flooding
        # while listening with nobody talking.
        return False

    # ── 4. CLIPPED AUDIO CHECK ──
    # Microphone saturation completely corrupts the STFT features
    if np.max(np.abs(audio_window)) >= CLIP_THRESHOLD:
        print("⚠️ [Stage 0] REJECTED: Microphone saturation / clipping detected. "
              "| Log: ERR_CLIP")
        return False

    # All 4 validation gates passed cleanly -> proceed to Stage 1 DSP
    return True