"""
Per-voice-channel state machine.

A `VoiceSession` owns:
- a `PcmSegmenter` that turns raw PCM frames into speech segments,
- a `WakeWordDetector` for the local pre-check,
- the per-channel `ConversationStore` handle,
- a status snapshot broadcast to the dashboard via a webhook callback.

The state machine is intentionally explicit:
    IDLE -> LISTEN -> STT -> THINK -> TTS -> SPEAK -> IDLE
    any state -> IDLE on interrupt / leave / fatal error
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from astrbot.api import logger

from .audio_pipeline import (
    TARGET_SAMPLE_RATE,
    change_speed,
    chunk_pcm_for_push,
    to_target_pcm,
    wav_to_pcm,
)
from .conversation_store import ConversationStore
from .provider_bridge import (
    clean_text_for_tts,
    llm_generate,
    split_long_text,
    stt_wav_to_text,
    tts_to_wav,
)
from .vad import PcmSegmenter, VadConfig, WakeWordDetector


class VoiceState(str, Enum):
    IDLE = "idle"
    LISTEN = "listen"
    STT = "stt"
    THINK = "think"
    TTS = "tts"
    SPEAK = "speak"
    ERROR = "error"


StatusCallback = Callable[[Dict[str, Any]], Awaitable[None]]
PushAudioCallback = Callable[[bytes], Awaitable[None]]


@dataclass
class SessionSnapshot:
    area_id: str
    channel_id: str
    state: VoiceState = VoiceState.IDLE
    last_text_in: str = ""
    last_text_out: str = ""
    last_error: str = ""
    last_active_at: float = field(default_factory=time.time)
    turn_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "area_id": self.area_id,
            "channel_id": self.channel_id,
            "key": f"{self.area_id}:{self.channel_id}",
            "state": self.state.value,
            "last_text_in": self.last_text_in,
            "last_text_out": self.last_text_out,
            "last_error": self.last_error,
            "last_active_at": self.last_active_at,
            "turn_count": self.turn_count,
            "extra": self.extra,
        }


class VoiceSession:
    """One session per OOPZ voice channel."""

    def __init__(
        self,
        area_id: str,
        channel_id: str,
        on_push_audio: PushAudioCallback,
        conversation: ConversationStore,
        *,
        stt_provider_id: str,
        tts_provider_id: str,
        llm_provider_id: str,
        wake_word: str,
        wake_variants: List[str],
        whisper_model: str,
        whisper_device: str,
        whisper_compute_type: str,
        whisper_language: str,
        whisper_enabled: bool,
        vad_config: VadConfig,
        tts_max_text_length: int,
        tts_split_long_text: bool,
        tts_speed: float,
        system_prompt: str,
        on_status: Optional[StatusCallback] = None,
        on_log: Optional[Callable[[str, str], Awaitable[None]]] = None,
        on_turn_complete: Optional[Callable[[str, str, str, str], Awaitable[None]]] = None,
    ) -> None:
        self.area_id = area_id
        self.channel_id = channel_id
        self._on_push_audio = on_push_audio
        self.conversation = conversation
        self.stt_provider_id = stt_provider_id
        self.tts_provider_id = tts_provider_id
        self.llm_provider_id = llm_provider_id
        self.system_prompt = system_prompt
        self._tts_max_text_length = tts_max_text_length
        self._tts_split_long_text = tts_split_long_text
        self._tts_speed = tts_speed
        self._vad_config = vad_config
        self._on_status = on_status
        self._on_log = on_log
        self._on_turn_complete = on_turn_complete

        self._segmenter = PcmSegmenter(vad_config)
        self._wake = WakeWordDetector(
            wake_word=wake_word,
            variants=wake_variants,
            model_size=whisper_model,
            device=whisper_device,
            compute_type=whisper_compute_type,
            language=whisper_language,
        ) if whisper_enabled else None
        self._whisper_enabled = whisper_enabled

        self._state = VoiceState.IDLE
        self._snapshot = SessionSnapshot(area_id=area_id, channel_id=channel_id)
        self._interrupt_event = asyncio.Event()
        self._wake_warmup_done = False
        self._lock = asyncio.Lock()
        self._current_turn_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def snapshot(self) -> SessionSnapshot:
        return self._snapshot

    @property
    def state(self) -> VoiceState:
        return self._state

    # ------------------------------------------------------------------
    # Runtime config updates
    # ------------------------------------------------------------------

    def update_tts_provider(self, provider_id: str) -> None:
        self.tts_provider_id = provider_id

    def update_stt_provider(self, provider_id: str) -> None:
        self.stt_provider_id = provider_id

    def update_llm_provider(self, provider_id: str) -> None:
        self.llm_provider_id = provider_id

    def update_wake_word(self, wake_word: str, variants: List[str]) -> None:
        if self._wake:
            self._wake.set_wake_word(wake_word, variants)
        self._vad_config.min_listen_ms = self._vad_config.min_listen_ms

    # ------------------------------------------------------------------
    # PCM ingestion
    # ------------------------------------------------------------------

    async def feed_pcm(self, pcm: bytes, user: str) -> None:
        if not pcm:
            return
        # We don't start a new turn if we're already mid-turn; let the active
        # task handle its current segment. We still update LISTEN state.
        async with self._lock:
            if self._state in (VoiceState.STT, VoiceState.THINK, VoiceState.TTS, VoiceState.SPEAK):
                # Skip this frame — we're already busy.
                return
            self._set_state(VoiceState.LISTEN, reason=f"user={user} feeding pcm")
        try:
            segments = await asyncio.get_running_loop().run_in_executor(
                None, self._segmenter.feed, pcm
            )
        except Exception as exc:
            logger.warning(f"[oopz] segmenter error: {exc}")
            return
        for seg in segments:
            await self._handle_segment(seg, user)

    async def flush(self) -> None:
        seg = await asyncio.get_running_loop().run_in_executor(None, self._segmenter.flush)
        if seg:
            await self._handle_segment(seg, user="?")

    # ------------------------------------------------------------------
    # External triggers
    # ------------------------------------------------------------------

    async def interrupt(self) -> None:
        self._interrupt_event.set()
        if self._current_turn_task and not self._current_turn_task.done():
            self._current_turn_task.cancel()
        self._set_state(VoiceState.IDLE, reason="interrupted")

    async def say(self, text: str) -> None:
        """Direct TTS bypass — speak `text` regardless of LISTEN/STT state."""
        # Cancel any ongoing playback so the user can be heard.
        await self.interrupt()
        task = asyncio.create_task(self._direct_say(text), name=f"oopz-say-{self.area_id}-{self.channel_id}")
        self._current_turn_task = task
        try:
            await task
        finally:
            self._current_turn_task = None

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    async def _handle_segment(self, pcm_segment: bytes, user: str) -> None:
        if not pcm_segment:
            return
        # Wake-word pre-check (optional)
        if self._wake is not None:
            try:
                if not self._wake_warmup_done:
                    await self._wake.warmup()
                    self._wake_warmup_done = True
                hit, transcript = await self._wake.detect(pcm_segment)
            except Exception as exc:
                logger.warning(f"[oopz] wake detector error: {exc}")
                hit, transcript = True, ""
            if not hit:
                # Not for us; reset and stay listening.
                self._set_state(VoiceState.LISTEN, reason="no wake word")
                return
        # Run the turn pipeline. Cancel any prior turn.
        if self._current_turn_task and not self._current_turn_task.done():
            self._current_turn_task.cancel()
        task = asyncio.create_task(
            self._run_turn(pcm_segment, user), name=f"oopz-turn-{self.area_id}-{self.channel_id}"
        )
        self._current_turn_task = task
        try:
            await task
        finally:
            self._current_turn_task = None

    async def _run_turn(self, pcm_segment: bytes, user: str) -> None:
        self._interrupt_event.clear()
        try:
            # 1) STT
            self._set_state(VoiceState.STT)
            wav_bytes = _pcm_to_wav(pcm_segment)
            text = await stt_wav_to_text(_context_or_none(self), self.stt_provider_id, wav_bytes)
            text = (text or "").strip()
            if not text:
                self._set_state(VoiceState.IDLE, reason="empty stt")
                return
            self._snapshot.last_text_in = text
            await self._log("user", text)
            await self.conversation.append(self.area_id, self.channel_id, "user", text)

            # 2) THINK
            self._set_state(VoiceState.THINK)
            history = await self.conversation.load(self.area_id, self.channel_id)
            # Remove the just-appended user turn from history to avoid duplication.
            if history and history[-1].get("role") == "user" and history[-1].get("content") == text:
                history = history[:-1]
            provider_id = self.llm_provider_id
            ctx = _context_or_none(self)
            if not provider_id:
                raise RuntimeError("LLM provider id is empty")
            response = await llm_generate(
                ctx, provider_id, self.system_prompt, history, text
            )
            response = (response or "").strip()
            if not response:
                self._set_state(VoiceState.IDLE, reason="empty llm")
                return
            self._snapshot.last_text_out = response
            self._snapshot.turn_count += 1
            await self._log("assistant", response)
            await self.conversation.append(self.area_id, self.channel_id, "assistant", response)
            # Push to AstrBot message pipeline so it appears in WebUI chat history
            if self._on_turn_complete is not None:
                try:
                    await self._on_turn_complete(self.area_id, self.channel_id, text, response)
                except Exception:
                    pass

            # 3) TTS
            await self._speak_text(response)
        except asyncio.CancelledError:
            logger.debug("[oopz] turn cancelled")
            self._set_state(VoiceState.IDLE, reason="cancelled")
            raise
        except Exception as exc:
            logger.error(f"[oopz] turn error: {exc}")
            self._snapshot.last_error = str(exc)
            self._set_state(VoiceState.ERROR, reason=str(exc))
            await asyncio.sleep(0.2)
            self._set_state(VoiceState.IDLE, reason="recovered")

    async def _direct_say(self, text: str) -> None:
        try:
            self._interrupt_event.clear()
            cleaned = clean_text_for_tts(text)
            if not cleaned:
                return
            self._snapshot.last_text_out = cleaned
            self._set_state(VoiceState.TTS, reason="direct say")
            await self._speak_text(cleaned)
        except asyncio.CancelledError:
            self._set_state(VoiceState.IDLE, reason="cancelled")
            raise
        except Exception as exc:
            logger.error(f"[oopz] direct_say error: {exc}")
            self._snapshot.last_error = str(exc)
            self._set_state(VoiceState.ERROR, reason=str(exc))
        finally:
            self._set_state(VoiceState.IDLE, reason="say done")

    async def _speak_text(self, text: str) -> None:
        cleaned = clean_text_for_tts(text)
        if not cleaned:
            return
        chunks: List[str]
        if self._tts_split_long_text and len(cleaned) > self._tts_max_text_length:
            chunks = split_long_text(cleaned, self._tts_max_text_length)
        else:
            chunks = [cleaned]
        for chunk in chunks:
            if self._interrupt_event.is_set():
                return
            self._set_state(VoiceState.TTS, reason=f"tts chunk len={len(chunk)}")
            ctx = _context_or_none(self)
            wav_bytes = await tts_to_wav(ctx, self.tts_provider_id, chunk)
            if not wav_bytes:
                continue
            await self._push_wav(wav_bytes)

    async def _push_wav(self, wav_bytes: bytes) -> None:
        try:
            pcm_info = wav_to_pcm(wav_bytes)
        except Exception as exc:
            logger.warning(f"[oopz] wav decode error: {exc}")
            return
        pcm = to_target_pcm(pcm_info)
        if self._tts_speed and abs(self._tts_speed - 1.0) > 1e-3:
            pcm = change_speed(pcm, self._tts_speed)
        # Re-wrap as WAV for the SDK voice backend
        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(pcm)
        wav_out = buf.getvalue()
        self._set_state(VoiceState.SPEAK, reason=f"push {len(wav_out)} bytes")
        try:
            await self._on_push_audio(wav_out)
        except Exception as exc:
            logger.error(f"[oopz] push_audio error: {exc}")
            self._snapshot.last_error = f"push_audio: {exc}"
        finally:
            self._set_state(VoiceState.IDLE, reason="speak done")

    # ------------------------------------------------------------------
    # Status plumbing
    # ------------------------------------------------------------------

    def _set_state(self, new_state: VoiceState, reason: str = "") -> None:
        if new_state == self._state:
            return
        self._state = new_state
        self._snapshot.state = new_state
        self._snapshot.last_active_at = time.time()
        if reason:
            self._snapshot.extra["last_reason"] = reason
        if self._on_status is not None:
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self._on_status(self._snapshot.to_dict()))
            except Exception:
                pass

    async def _log(self, role: str, text: str) -> None:
        if self._on_log is None:
            return
        try:
            await self._on_log(role, text)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcm_to_wav(pcm: bytes) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


def _context_or_none(session: "VoiceSession") -> Any:
    """Return the AstrBot `Context` attached to this session, or a stub.

    We stash the real Context on the session via the `oopz` reference's owner.
    For backward compatibility we also check a private attribute.
    """
    ctx = getattr(session, "_context", None)
    if ctx is not None:
        return ctx
    return _ContextHolder.default


class _ContextHolder:
    default: Any = None
