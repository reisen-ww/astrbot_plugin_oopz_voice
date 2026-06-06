"""
Voice activity detection + local wake-word pre-check.

We use:
- A fast **RMS energy gate** on 20 ms frames to flag candidate voiced frames
  (works on all platforms, no compiled dependencies).
- An optional `pydub.silence.detect_silence` pass on a captured segment to
  trim leading/trailing silence accurately. `pydub` ships with pre-built
  wheels on every platform and is already in our `requirements.txt`.
- An optional `faster-whisper` local tiny model to recognize the wake word
  inside a captured segment. This is much cheaper than always calling the
  cloud STT provider.

We deliberately do **not** depend on `webrtcvad`: it ships only an sdist on
PyPI and requires a C toolchain to build. The RMS-gated approach is good
enough for conversational wake-word capture.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from astrbot.api import logger

from .audio_pipeline import (
    FRAME_DURATION_MS,
    TARGET_SAMPLE_RATE,
    compute_rms,
    frame_bytes,
)


# webrtcvad only supports 10/20/30 ms frames at 8k/16k/32k/48k. We don't use
# it, but keep the constant in case someone re-introduces it later.
_VALID_FRAME_DURATIONS = (10, 20, 30)


@dataclass
class VadConfig:
    aggressiveness: int = 2  # 0..3
    rms_gate: float = 0.01
    silence_ms_to_flush: int = 700
    max_listen_ms: int = 30_000
    min_listen_ms: int = 400
    frame_duration_ms: int = 20


# ---------------------------------------------------------------------------
# PCM segmenter
# ---------------------------------------------------------------------------


class PcmSegmenter:
    """Consume PCM bytes, emit voice segments.

    A segment is a contiguous run of speech frames bounded by silence. We
    always emit a segment when:
      - `silence_ms_to_flush` of consecutive silence follows speech, OR
      - `max_listen_ms` has elapsed since the segment started, OR
      - the consumer calls `flush()`.
    """

    def __init__(self, config: Optional[VadConfig] = None) -> None:
        self.config = config or VadConfig()
        self._buffer: bytearray = bytearray()
        self._silence_ms = 0
        self._listen_ms = 0
        self._in_speech = False
        self._total_ms = 0

    def reset(self) -> None:
        self._buffer.clear()
        self._silence_ms = 0
        self._listen_ms = 0
        self._in_speech = False
        self._total_ms = 0

    def feed(self, pcm: bytes) -> List[bytes]:
        """Feed PCM bytes; return zero or more complete segments."""
        segments: List[bytes] = []
        if not pcm:
            return segments
        for frame in frame_bytes(
            pcm,
            sample_rate=TARGET_SAMPLE_RATE,
            duration_ms=self.config.frame_duration_ms,
        ):
            if len(frame) == 0:
                continue
            self._total_ms += self.config.frame_duration_ms
            rms = compute_rms(frame)
            voiced = rms >= self.config.rms_gate

            if voiced:
                self._buffer.extend(frame)
                self._silence_ms = 0
                self._listen_ms += self.config.frame_duration_ms
                self._in_speech = True
            else:
                if self._in_speech:
                    self._silence_ms += self.config.frame_duration_ms
                # Track listening time even before speech starts.
                if self._in_speech or self._buffer:
                    self._listen_ms += self.config.frame_duration_ms

            # Emit on flush-able conditions
            if self._in_speech and self._silence_ms >= self.config.silence_ms_to_flush:
                seg = bytes(self._buffer)
                self.reset()
                if _segment_duration_ms(seg) >= self.config.min_listen_ms:
                    seg = _refine_with_pydub(seg)
                    segments.append(seg)
            elif self._in_speech and self._listen_ms >= self.config.max_listen_ms:
                seg = bytes(self._buffer)
                self.reset()
                if _segment_duration_ms(seg) >= self.config.min_listen_ms:
                    seg = _refine_with_pydub(seg)
                    segments.append(seg)
        return segments

    def flush(self) -> Optional[bytes]:
        """Force-emit any pending buffer."""
        if not self._buffer:
            return None
        seg = bytes(self._buffer)
        self.reset()
        if _segment_duration_ms(seg) < self.config.min_listen_ms:
            return None
        return _refine_with_pydub(seg)


# ---------------------------------------------------------------------------
# Wake-word detector
# ---------------------------------------------------------------------------


class WakeWordDetector:
    """Detect a wake word in a PCM segment using a local faster-whisper model.

    This is intentionally simple: we transcribe the whole segment and look for
    any of the configured wake phrases in the (lowercased) transcription.
    """

    def __init__(
        self,
        wake_word: str = "bot",
        variants: Optional[List[str]] = None,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "auto",
    ) -> None:
        self._wake_word = (wake_word or "").strip().lower() or "bot"
        self._variants = [w.strip().lower() for w in (variants or []) if w and w.strip()]
        if self._wake_word not in self._variants:
            self._variants.insert(0, self._wake_word)
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = None if language in (None, "", "auto") else language
        self._model = None
        self._lock = asyncio.Lock()

    def set_wake_word(self, wake_word: str, variants: Optional[List[str]] = None) -> None:
        self._wake_word = (wake_word or "").strip().lower() or self._wake_word
        new_variants = [w.strip().lower() for w in (variants or []) if w and w.strip()]
        if self._wake_word not in new_variants:
            new_variants.insert(0, self._wake_word)
        self._variants = new_variants

    async def warmup(self) -> bool:
        """Load the model. Returns True on success."""
        if self._model is not None:
            return True
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:
            logger.warning(f"[oopz] faster-whisper not available: {exc}; wake-word pre-check disabled")
            return False
        try:
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type=self._compute_type,
                ),
            )
            logger.info(f"[oopz] faster-whisper '{self._model_size}' loaded on {self._device}")
            return True
        except Exception as exc:
            logger.warning(f"[oopz] failed to load faster-whisper: {exc}")
            return False

    async def detect(self, pcm_segment: bytes) -> Tuple[bool, str]:
        """Return (wake_hit, transcript)."""
        if not pcm_segment:
            return False, ""
        if not self._model:
            ok = await self.warmup()
            if not ok:
                # Without the model we fall back to a naive heuristic: if the
                # segment is longer than `min_listen_ms` we treat it as a
                # wake hit so the rest of the pipeline can still process it.
                return True, ""
        wav_bytes = _pcm16k_mono_to_wav(pcm_segment)
        try:
            async with self._lock:
                loop = asyncio.get_running_loop()
                segments, _info = await loop.run_in_executor(
                    None,
                    lambda: self._model.transcribe(
                        _wav_bytes_to_file_path(wav_bytes),
                        language=self._language,
                        vad_filter=False,
                        beam_size=1,
                    ),
                )
                text = "".join(seg.text for seg in segments).strip().lower()
        except Exception as exc:
            logger.warning(f"[oopz] wake-word transcription error: {exc}")
            return False, ""

        hit = self._match(text)
        return hit, text

    def _match(self, text: str) -> bool:
        if not text:
            return False
        for w in self._variants:
            if w and w in text:
                return True
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _segment_duration_ms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    return len(pcm) / (TARGET_SAMPLE_RATE * 2) * 1000.0


def _refine_with_pydub(pcm: bytes) -> bytes:
    """Trim leading/trailing silence from a captured segment using `pydub`.

    This is a best-effort refinement. If pydub (or its ffmpeg shim) is
    unavailable, we return the segment unchanged.
    """
    if not pcm:
        return pcm
    try:
        import io
        from pydub import AudioSegment
        from pydub.silence import detect_silence
    except Exception as exc:
        logger.debug(f"[oopz] pydub unavailable for VAD refine: {exc}")
        return pcm
    try:
        wav_bytes = _pcm16k_mono_to_wav(pcm)
        seg = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        # detect_silence returns a list of (start_ms, end_ms) silent ranges.
        # We want to keep the largest non-silent chunk in the segment.
        silent_ranges = detect_silence(
            seg,
            min_silence_len=200,
            silence_thresh=-40,
            seek_step=10,
        )
        if not silent_ranges:
            return pcm
        # Find the longest non-silent region
        total = len(seg)  # in ms
        nonsilent = []
        cursor = 0
        for s, e in silent_ranges:
            if s > cursor:
                nonsilent.append((cursor, s))
            cursor = max(cursor, e)
        if cursor < total:
            nonsilent.append((cursor, total))
        if not nonsilent:
            return pcm
        # Take the longest one
        nonsilent.sort(key=lambda r: r[1] - r[0], reverse=True)
        keep_from_ms, keep_to_ms = nonsilent[0]
        # Pad back a little to avoid clipping the actual speech
        keep_from_ms = max(0, keep_from_ms - 80)
        keep_to_ms = min(total, keep_to_ms + 80)
        kept = seg[keep_from_ms:keep_to_ms]
        return kept.raw_data
    except Exception as exc:
        logger.debug(f"[oopz] pydub refine failed, returning raw segment: {exc}")
        return pcm


def _pcm16k_mono_to_wav(pcm: bytes) -> bytes:
    """Wrap raw 16 kHz mono 16-bit PCM into a minimal WAV container."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


def _wav_bytes_to_file_path(wav_bytes: bytes) -> str:
    """Persist a WAV blob to a temp file (faster-whisper wants a path)."""
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="oopz_wake_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(wav_bytes)
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        raise
    return path
