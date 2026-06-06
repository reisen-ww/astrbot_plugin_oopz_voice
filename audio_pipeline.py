"""
Audio conversion / resampling utilities.

The OOPZ voice stream arrives as raw PCM frames whose sample rate and channel
layout may differ across SDK versions / platforms. The TTS provider returns
WAV bytes that we need to feed back as PCM frames at the right rate.

We deliberately keep dependencies light: numpy + the Python `wave` module.
For non-trivial resampling (e.g. 24 kHz → 16 kHz) we fall back to a small
linear-interpolation routine that's good enough for speech.

For higher quality resampling we can plug in `scipy.signal.resample_poly` if
it's available; otherwise we degrade gracefully.
"""
from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from astrbot.api import logger


# 20 ms of 16 kHz mono PCM is the canonical frame size for webrtcvad.
FRAME_DURATION_MS = 20
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # 16-bit


@dataclass
class PcmInfo:
    sample_rate: int
    channels: int
    sample_width: int
    pcm: bytes


# ---------------------------------------------------------------------------
# Frame iteration
# ---------------------------------------------------------------------------


def frame_bytes(pcm: bytes, sample_rate: int = TARGET_SAMPLE_RATE,
                duration_ms: int = FRAME_DURATION_MS) -> Iterable[bytes]:
    """Yield successive fixed-size PCM frames."""
    frame_size = int(sample_rate * (duration_ms / 1000.0)) * TARGET_SAMPLE_WIDTH * TARGET_CHANNELS
    if frame_size <= 0:
        return
    for i in range(0, len(pcm), frame_size):
        chunk = pcm[i:i + frame_size]
        if len(chunk) == frame_size:
            yield chunk


def pcm_duration_ms(pcm: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
    """Return the duration of a PCM byte string in milliseconds."""
    bytes_per_ms = sample_rate * TARGET_SAMPLE_WIDTH * TARGET_CHANNELS / 1000.0
    if bytes_per_ms <= 0:
        return 0.0
    return len(pcm) / bytes_per_ms


# ---------------------------------------------------------------------------
# WAV <-> PCM
# ---------------------------------------------------------------------------


def wav_to_pcm(wav_bytes: bytes) -> PcmInfo:
    """Decode a WAV byte string into raw PCM plus its parameters."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return PcmInfo(
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        pcm=pcm,
    )


def pcm_to_wav(pcm: PcmInfo) -> bytes:
    """Wrap raw PCM into a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(pcm.channels)
        wf.setsampwidth(pcm.sample_width)
        wf.setframerate(pcm.sample_rate)
        wf.writeframes(pcm.pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Conversion to canonical 16 kHz mono 16-bit
# ---------------------------------------------------------------------------


def to_target_pcm(pcm_info: PcmInfo,
                  target_rate: int = TARGET_SAMPLE_RATE,
                  target_channels: int = TARGET_CHANNELS,
                  target_sample_width: int = TARGET_SAMPLE_WIDTH) -> bytes:
    """Convert arbitrary PCM to a canonical mono 16-bit layout at `target_rate`."""
    if pcm_info.sample_width != 2:
        raise ValueError(
            f"Only 16-bit PCM is supported for resampling (got width={pcm_info.sample_width})"
        )
    if len(pcm_info.pcm) == 0:
        return b""

    audio = np.frombuffer(pcm_info.pcm, dtype=np.int16)
    if pcm_info.channels > 1:
        audio = audio.reshape(-1, pcm_info.channels).mean(axis=1).astype(np.int16)
    if pcm_info.sample_rate != target_rate:
        audio = resample_audio(audio, pcm_info.sample_rate, target_rate).astype(np.int16)
    if audio.dtype != np.int16:
        audio = audio.astype(np.int16)
    return audio.tobytes()


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a 1-D float/int array. Uses scipy if available, else linear."""
    if src_rate == dst_rate or len(audio) == 0:
        return audio

    try:
        from scipy.signal import resample_poly  # type: ignore
        from math import gcd

        g = gcd(src_rate, dst_rate)
        up = dst_rate // g
        down = src_rate // g
        out = resample_poly(audio.astype(np.float32), up, down)
        return out
    except Exception:
        # Fall back to linear interpolation — good enough for short TTS chunks.
        duration = len(audio) / float(src_rate)
        dst_len = int(round(duration * dst_rate))
        if dst_len <= 1:
            return np.array([audio[0]], dtype=audio.dtype)
        src_x = np.linspace(0.0, duration, num=len(audio), endpoint=True)
        dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=True)
        return np.interp(dst_x, src_x, audio.astype(np.float32)).astype(audio.dtype)


# ---------------------------------------------------------------------------
# RMS / simple VAD utilities
# ---------------------------------------------------------------------------


def compute_rms(pcm: bytes) -> float:
    """Return the RMS amplitude of a 16-bit PCM buffer in [0, 1]."""
    if not pcm:
        return 0.0
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)) / 32768.0)


def is_silent(pcm: bytes, rms_gate: float) -> bool:
    return compute_rms(pcm) < rms_gate


# ---------------------------------------------------------------------------
# Speed / pitch (very simple linear interpolation; speech only)
# ---------------------------------------------------------------------------


def change_speed(pcm: bytes, speed: float) -> bytes:
    """Resample-and-trim a 16-bit mono PCM buffer to simulate playback speed."""
    if not pcm or speed <= 0 or abs(speed - 1.0) < 1e-3:
        return pcm
    arr = np.frombuffer(pcm, dtype=np.int16)
    new_len = max(1, int(len(arr) / speed))
    src_x = np.linspace(0.0, 1.0, num=len(arr), endpoint=True)
    dst_x = np.linspace(0.0, 1.0, num=new_len, endpoint=True)
    out = np.interp(dst_x, src_x, arr.astype(np.float32)).astype(np.int16)
    return out.tobytes()


# ---------------------------------------------------------------------------
# Long-audio chunking
# ---------------------------------------------------------------------------


def chunk_pcm_for_push(pcm: bytes, chunk_ms: int = 100) -> List[bytes]:
    """Split a PCM byte string into fixed-duration chunks (in milliseconds)."""
    if not pcm:
        return []
    bytes_per_ms = TARGET_SAMPLE_RATE * TARGET_SAMPLE_WIDTH * TARGET_CHANNELS / 1000.0
    chunk_size = max(1, int(bytes_per_ms * chunk_ms))
    return [pcm[i:i + chunk_size] for i in range(0, len(pcm), chunk_size) if pcm[i:i + chunk_size]]
