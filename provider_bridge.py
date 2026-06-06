"""
Provider bridge for AstrBot TTS / STT / LLM providers.

This module abstracts over the various Provider implementations shipped with
AstrBot. Different Provider types expose slightly different methods; we probe
with `hasattr` and fall back gracefully.

Important: when calling `llm_generate` directly we must sanitize the contexts
list — see the project's `references/mistake-book.md` (OpenAI Responses
"Missing required parameter: input[x].call_id" issue).
"""
from __future__ import annotations

import asyncio
import io
import re
import wave
from typing import Any, Iterable, List, Optional

from astrbot.api import logger
from astrbot.api.star import Context


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


async def get_tts_provider(context: Context, provider_id: str) -> Any:
    """Resolve a TTS provider by id and validate its type."""
    if not provider_id:
        raise ValueError("TTS provider id is empty")
    provider = await context.get_provider_by_id(provider_id)
    if provider is None:
        raise ValueError(f"TTS provider not found: {provider_id}")
    meta = _provider_meta(provider)
    if meta and meta != "text_to_speech":
        raise ValueError(
            f"Provider {provider_id} is not a text_to_speech provider (got {meta})"
        )
    return provider


async def get_stt_provider(context: Context, provider_id: str) -> Any:
    """Resolve an STT provider by id and validate its type."""
    if not provider_id:
        raise ValueError("STT provider id is empty")
    provider = await context.get_provider_by_id(provider_id)
    if provider is None:
        raise ValueError(f"STT provider not found: {provider_id}")
    meta = _provider_meta(provider)
    if meta and meta != "speech_to_text":
        raise ValueError(
            f"Provider {provider_id} is not a speech_to_text provider (got {meta})"
        )
    return provider


async def get_llm_provider_id(context: Context, umo: str, preferred: Optional[str]) -> str:
    """Resolve LLM provider id. Falls back to the provider bound to `umo`."""
    if preferred:
        return preferred
    pid = await context.get_current_chat_provider_id(umo=umo)
    if not pid:
        raise ValueError("No LLM provider configured for the current session")
    return pid


def _provider_meta(provider: Any) -> Optional[str]:
    """Best-effort lookup of a Provider's `provider_type` field."""
    for attr in ("provider_type", "type", "meta"):
        if hasattr(provider, attr):
            value = getattr(provider, attr)
            if isinstance(value, str):
                return value
            if hasattr(value, "provider_type"):
                return getattr(value, "provider_type")
    return None


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


async def tts_to_wav(
    context: Context,
    provider_id: str,
    text: str,
) -> bytes:
    """Call a TTS provider and return WAV bytes.

    Different providers expose different methods; we try the most common ones.
    """
    if not text.strip():
        return b""

    provider = await get_tts_provider(context, provider_id)
    coro = _call_provider_tts(provider, text)
    wav_bytes = await coro
    if not isinstance(wav_bytes, (bytes, bytearray)):
        raise RuntimeError(
            f"TTS provider {provider_id} returned {type(wav_bytes).__name__}, expected bytes"
        )
    return bytes(wav_bytes)


async def _call_provider_tts(provider: Any, text: str) -> bytes:
    """Try several TTS method names exposed by the provider."""
    method_candidates = [
        "get_audio",
        "text_to_speech",
        "tts",
        "synthesize",
    ]
    for name in method_candidates:
        if not hasattr(provider, name):
            continue
        method = getattr(provider, name)
        if not callable(method):
            continue
        result = method(text)
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, (bytes, bytearray)):
            return bytes(result)
        # Some providers return a dict with `audio` / `data` / `url`.
        if isinstance(result, dict):
            for key in ("audio", "data", "audio_data", "wav", "result"):
                if key in result and isinstance(result[key], (bytes, bytearray)):
                    return bytes(result[key])
            for key in ("url", "audio_url"):
                if key in result and isinstance(result[key], str):
                    return await _download_to_bytes(result[key])
    raise RuntimeError(
        f"TTS provider {type(provider).__name__} exposes no known TTS method"
    )


async def _download_to_bytes(url: str) -> bytes:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            return await resp.read()


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------


async def stt_wav_to_text(
    context: Context,
    provider_id: str,
    wav_bytes: bytes,
) -> str:
    """Call an STT provider and return transcribed text."""
    if not wav_bytes:
        return ""
    provider = await get_stt_provider(context, provider_id)
    return await _call_provider_stt(provider, wav_bytes)


async def _call_provider_stt(provider: Any, wav_bytes: bytes) -> str:
    """Try several STT method names exposed by the provider."""
    method_candidates = [
        "get_text",
        "speech_to_text",
        "stt",
        "transcribe",
        "recognize",
    ]
    for name in method_candidates:
        if not hasattr(provider, name):
            continue
        method = getattr(provider, name)
        if not callable(method):
            continue
        result = method(wav_bytes)
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            for key in ("text", "result", "transcript", "transcription"):
                if key in result and isinstance(result[key], str):
                    return result[key].strip()
    raise RuntimeError(
        f"STT provider {type(provider).__name__} exposes no known STT method"
    )


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


async def llm_generate(
    context: Context,
    provider_id: str,
    system_prompt: str,
    history: List[dict],
    user_text: str,
) -> str:
    """Call the chat provider and return its text response.

    We sanitize `history` so it only contains plain text messages with the
    `system` / `user` / `assistant` roles. This avoids OpenAI Responses API
    complaints about half-finished tool call chains.
    """
    sanitized = _sanitize_context(history)
    sanitized.append({"role": "user", "content": user_text})

    try:
        resp = await context.llm_generate(
            chat_provider_id=provider_id,
            prompt=user_text,
            system_prompt=system_prompt,
            contexts=sanitized,
        )
    except TypeError:
        # Older AstrBot versions might not accept `contexts` kwarg.
        resp = await context.llm_generate(
            chat_provider_id=provider_id,
            prompt=user_text,
            system_prompt=system_prompt,
        )
    return _extract_text(resp)


def _sanitize_context(history: Iterable[dict]) -> List[dict]:
    """Keep only plain system / user / assistant messages."""
    out: List[dict] = []
    for msg in history or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for seg in content:
                if isinstance(seg, dict) and seg.get("type") in (None, "text"):
                    parts.append(str(seg.get("text", "")))
                elif isinstance(seg, str):
                    parts.append(seg)
            content = "".join(parts)
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def _extract_text(resp: Any) -> str:
    """Extract the plain text from a variety of LLM response shapes."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp.strip()
    for attr in ("completion_text", "text", "content", "result"):
        if hasattr(resp, attr):
            value = getattr(resp, attr)
            if isinstance(value, str):
                return value.strip()
    if isinstance(resp, dict):
        for key in ("completion_text", "text", "content", "result"):
            if key in resp and isinstance(resp[key], str):
                return resp[key].strip()
    return str(resp).strip()


# ---------------------------------------------------------------------------
# Audio utilities used by both TTS and STT
# ---------------------------------------------------------------------------


def wav_to_pcm16k_mono(wav_bytes: bytes) -> bytes:
    """Decode WAV bytes to 16 kHz mono 16-bit PCM."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    if channels == 1 and sample_rate == 16000:
        return raw

    import numpy as np

    audio = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if sample_rate != 16000:
        audio = _resample_linear(audio, sample_rate, 16000).astype(np.int16)
    return audio.tobytes()


def pcm16k_mono_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw PCM into a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _resample_linear(audio, src_rate: int, dst_rate: int):
    import numpy as np

    if src_rate == dst_rate or len(audio) == 0:
        return audio
    duration = len(audio) / float(src_rate)
    dst_len = int(round(duration * dst_rate))
    if dst_len <= 1:
        return np.array([audio[0]], dtype=audio.dtype)
    src_x = np.linspace(0, duration, num=len(audio), endpoint=True)
    dst_x = np.linspace(0, duration, num=dst_len, endpoint=True)
    return np.interp(dst_x, src_x, audio.astype(np.float32)).astype(audio.dtype)


# ---------------------------------------------------------------------------
# Text cleanup before TTS
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HEADING_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_BOLD_ITALIC_RE = re.compile(r"(\*\*|__|\*|_)(.*?)\1")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*")


def clean_text_for_tts(text: str) -> str:
    """Strip markdown and other visual noise that shouldn't be spoken."""
    if not text:
        return ""
    s = text
    s = _FENCE_RE.sub("", s)
    s = _INLINE_CODE_RE.sub("", s)
    s = _MARKDOWN_LINK_RE.sub(r"\1", s)
    s = _HEADING_RE.sub("", s)
    s = _BOLD_ITALIC_RE.sub(r"\2", s)
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_long_text(text: str, max_length: int) -> List[str]:
    """Split a long string into chunks no longer than `max_length` characters."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_length:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_length:
        head = remaining[:max_length]
        cut = max(head.rfind("。"), head.rfind("！"), head.rfind("？"),
                  head.rfind("."), head.rfind("!"), head.rfind("?"), head.rfind(";"))
        if cut < max_length * 0.5:
            cut = max_length
        chunks.append(remaining[:cut + 1].strip())
        remaining = remaining[cut + 1:].strip()
    if remaining:
        chunks.append(remaining)
    return [c for c in chunks if c]
