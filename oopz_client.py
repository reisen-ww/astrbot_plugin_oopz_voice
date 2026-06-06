"""
Thin async wrapper around `oopz_sdk` (v0.13.x).

This module is deliberately tolerant: the OOPZ SDK has many version-specific
quirks and the public surface for voice streaming is still evolving. We probe
for the actual attributes that exist on the installed SDK and fail loud only
when something is *definitively* missing.

Verified against `oopz-sdk 0.13.1`:
  - `OopzBot(config, on_ready=..., on_raw_event=..., on_message=...)`
  - `bot.rest.channels` : channels service (info)
  - `bot.rest.voice` / `bot.voice` : voice service
  - `voice.join(area=..., channel=...)` -> ChannelSign
  - `voice.leave()`
  - `voice.play_bytes(data: bytes, *, mime_type='audio/mpeg')`
  - `voice.pause()` / `voice.resume()` / `voice.stop()`
  - `bot.on_raw_event` (decorator) for raw event stream
  - `bot.on_message`, `bot.on_ready` (decorators)
"""
from __future__ import annotations

import asyncio
import base64
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from astrbot.api import logger

try:
    from oopz_sdk import OopzBot, OopzConfig  # type: ignore
    _OOPZ_AVAILABLE = True
    _IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - depends on env
    OopzBot = None  # type: ignore
    OopzConfig = None  # type: ignore
    _OOPZ_AVAILABLE = False
    _IMPORT_ERROR = str(exc)


AudioCallback = Callable[[str, str, str, bytes], Awaitable[None]]
VoiceEventCallback = Callable[[Dict[str, Any]], Awaitable[None]]


# ---------------------------------------------------------------------------
# Audio-event detection
# ---------------------------------------------------------------------------
#
# The SDK does not expose a dedicated `on_audio_frame` hook. Voice frames come
# in through the raw WebSocket event stream (`on_raw_event`). We sniff for
# voice-related events there. The exact event shape is not part of the public
# SDK API; we accept a few common field names.

_AUDIO_EVENT_HINTS = (
    "audio", "voice", "frame", "pcm", "opus", "rtc", "track",
)
_AREA_KEYS = ("area_id", "area", "areaId")
_CHANNEL_KEYS = ("channel_id", "channel", "channelId")
_USER_KEYS = ("user_id", "user", "sender", "uid", "userId")
_PCM_KEYS = ("pcm", "data", "audio", "frame", "samples", "payload")


@dataclass
class OopzAuth:
    device_id: str = ""
    person_uid: str = ""
    jwt_token: str = ""
    private_key: str = ""


@dataclass
class JoinedChannel:
    area_id: str
    channel_id: str


@dataclass
class OopzClientStatus:
    connected: bool = False
    ready: bool = False
    joined: List[JoinedChannel] = field(default_factory=list)
    last_error: Optional[str] = None
    sdk_available: bool = _OOPZ_AVAILABLE
    sdk_import_error: Optional[str] = _IMPORT_ERROR


# ---------------------------------------------------------------------------
# Auto-install helper
# ---------------------------------------------------------------------------


async def ensure_oopz_sdk_installed(pip_install_timeout: int = 180) -> bool:
    """Install `oopz-sdk` via pip with `--no-deps`.

    This bypasses AstrBot's cryptography-version protection. The 0.13.x SDK
    pins `cryptography<48` but AstrBot core ships with `cryptography==48.0.0`.
    The 48.x API is backwards-compatible with what oopz-sdk uses internally
    (RSA PKCS#1 v1.5 + SHA-256 signing of REST requests).

    Returns True if `oopz_sdk` is importable after the attempt.
    """
    global _OOPZ_AVAILABLE, _IMPORT_ERROR, OopzBot, OopzConfig

    if _OOPZ_AVAILABLE:
        return True

    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--disable-pip-version-check", "oopz-sdk"],
            capture_output=True,
            text=True,
            timeout=pip_install_timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error("[oopz] pip install oopz-sdk timed out")
        return False
    except Exception as exc:
        logger.error(f"[oopz] pip install oopz-sdk failed: {exc}")
        return False

    if proc.returncode != 0:
        logger.warning(
            f"[oopz] pip install oopz-sdk returned {proc.returncode}: "
            f"{(proc.stderr or proc.stdout).strip()[:300]}"
        )

    try:
        import importlib
        # Force a fresh import — if the package was added to a path that
        # the running interpreter hasn't scanned, importlib.reload is needed.
        if "oopz_sdk" in sys.modules:
            importlib.reload(sys.modules["oopz_sdk"])
        mod = importlib.import_module("oopz_sdk")
        OopzBot = getattr(mod, "OopzBot", None)
        OopzConfig = getattr(mod, "OopzConfig", None)
        if OopzBot is not None and OopzConfig is not None:
            _OOPZ_AVAILABLE = True
            _IMPORT_ERROR = None
            logger.info("[oopz] oopz-sdk installed and importable")
            return True
    except Exception as exc:
        _IMPORT_ERROR = str(exc)
        logger.warning(f"[oopz] oopz-sdk still not importable after pip install: {exc}")

    logger.info(
        "[oopz] oopz-sdk was installed but is not yet importable in this process. "
        "Please click 'Reload plugin' in the WebUI to pick it up."
    )
    return False


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OopzClient:
    """Async wrapper around the OOPZ SDK."""

    def __init__(self, auth: OopzAuth) -> None:
        self._auth = auth
        self._bot: Any = None
        self._rest: Any = None
        self._voice: Any = None
        self._connected = False
        self._ready = False
        self._joined: Set[tuple[str, str]] = set()
        self._audio_cbs: List[AudioCallback] = []
        self._voice_evt_cbs: List[VoiceEventCallback] = []
        self._run_task: Optional[asyncio.Task] = None
        self._last_error: Optional[str] = None
        self._stopped = asyncio.Event()
        self._reconnect_delay = 5.0

    # ------------------------------------------------------------------
    # Public status
    # ------------------------------------------------------------------

    @property
    def status(self) -> OopzClientStatus:
        return OopzClientStatus(
            connected=self._connected,
            ready=self._ready,
            joined=[JoinedChannel(a, c) for a, c in sorted(self._joined)],
            last_error=self._last_error,
            sdk_available=_OOPZ_AVAILABLE,
            sdk_import_error=_IMPORT_ERROR,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not _OOPZ_AVAILABLE:
            raise RuntimeError(
                f"oopz-sdk is not importable: {_IMPORT_ERROR}. "
                "Install it with `pip install oopz-sdk --no-deps`."
            )
        if self._run_task and not self._run_task.done():
            logger.info("[oopz] client already running")
            return

        self._stopped.clear()
        self._run_task = asyncio.create_task(self._run_forever(), name="oopz-client")

    async def stop(self) -> None:
        self._stopped.set()
        if self._bot is not None:
            try:
                stop = getattr(self._bot, "stop", None)
                if stop is not None:
                    result = stop()
                    if inspect.iscoroutine(result):
                        await result
            except Exception as exc:
                logger.warning(f"[oopz] bot.stop() error: {exc}")
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except (asyncio.CancelledError, Exception):
                pass
        self._run_task = None
        self._bot = None
        self._rest = None
        self._voice = None
        self._connected = False
        self._ready = False
        self._joined.clear()

    async def _run_forever(self) -> None:
        backoff = self._reconnect_delay
        while not self._stopped.is_set():
            try:
                await self._connect_once()
                backoff = self._reconnect_delay
                while not self._stopped.is_set() and self._connected:
                    await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = repr(exc)
                logger.error(f"[oopz] connection error: {exc}")
            finally:
                self._connected = False
                self._ready = False
            if self._stopped.is_set():
                break
            logger.info(f"[oopz] reconnecting in {backoff:.1f}s")
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, 60.0)

    async def _connect_once(self) -> None:
        cfg = OopzConfig(
            device_id=self._auth.device_id,
            person_uid=self._auth.person_uid,
            jwt_token=self._auth.jwt_token,
            private_key=self._auth.private_key,
        )
        bot = OopzBot(
            cfg,
            on_ready=self._on_ready,
            on_close=self._on_disconnect,
            on_error=self._on_error,
            on_raw_event=self._on_raw_event,
            on_message=self._on_message_log,
        )
        self._bot = bot
        self._rest = getattr(bot, "rest", None)
        self._voice = getattr(bot, "voice", None) or _safe_get(self._rest, "voice")

        # Register hook-style handlers too (some SDK versions prefer them).
        for evt in ("on_voice_event", "on_voice_frame", "on_audio_frame"):
            _register_decorator(bot, evt, getattr(self, "_" + evt, self._on_voice_event))

        self._connected = True
        logger.info("[oopz] bot.run() starting")
        run = bot.run()
        if inspect.iscoroutine(run):
            await run
        else:
            # Some SDK versions make run() blocking; give the loop a beat.
            while not self._stopped.is_set() and self._connected:
                await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # Built-in handlers
    # ------------------------------------------------------------------

    async def _on_ready(self, *_args, **_kwargs) -> None:
        self._ready = True
        logger.info("[oopz] ready")
        for area, channel in list(self._joined):
            try:
                await self.join_voice(area, channel)
            except Exception as exc:
                logger.warning(f"[oopz] re-join {area}/{channel} failed: {exc}")

    async def _on_disconnect(self, *_args, **_kwargs) -> None:
        self._ready = False
        logger.warning("[oopz] disconnected")

    async def _on_error(self, *_args, **_kwargs) -> None:
        self._last_error = str(_args[0] if _args else _kwargs)
        logger.error(f"[oopz] on_error: {self._last_error}")

    async def _on_message_log(self, message: Any, *_args, **_kwargs) -> None:
        # Text messages are deliberately NOT bridged to AstrBot; just log.
        try:
            text = getattr(message, "content", None) or getattr(message, "text", None) or str(message)
            sender = getattr(message, "sender", None)
            sender_name = getattr(sender, "nickname", None) or getattr(sender, "user_id", "?")
            area = getattr(message, "area_id", "?")
            channel = getattr(message, "channel_id", "?")
            logger.debug(f"[oopz] text-only (ignored) {area}/{channel} {sender_name}: {text}")
        except Exception:
            pass

    async def _on_raw_event(self, event: Any) -> None:
        """The SDK's raw WS event hook. We sniff for voice frames here."""
        try:
            data = _to_dict(event)
        except Exception:
            data = {"raw": str(event)}
        # If this looks like a voice/audio event, try to extract PCM and
        # dispatch to subscribers.
        try:
            if _looks_like_audio_event(data):
                area, channel, user, pcm = _extract_audio_frame(data)
                if pcm:
                    for cb in self._audio_cbs:
                        try:
                            await cb(area, channel, user, pcm)
                        except Exception as exc:
                            logger.warning(f"[oopz] audio callback error: {exc}")
        except Exception as exc:
            logger.debug(f"[oopz] raw event parse error: {exc}")
        # Forward to any voice event subscribers (for logging/UI).
        for cb in self._voice_evt_cbs:
            try:
                await cb(data)
            except Exception as exc:
                logger.warning(f"[oopz] voice event callback error: {exc}")

    async def _on_voice_event(self, event: Any) -> None:
        try:
            data = _to_dict(event)
        except Exception:
            data = {"raw": str(event)}
        logger.debug(f"[oopz] voice event: {data}")
        for cb in self._voice_evt_cbs:
            try:
                await cb(data)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def subscribe_audio(self, callback: AudioCallback) -> None:
        self._audio_cbs.append(callback)

    def subscribe_voice_event(self, callback: VoiceEventCallback) -> None:
        self._voice_evt_cbs.append(callback)

    # ------------------------------------------------------------------
    # Voice channel control
    # ------------------------------------------------------------------

    async def join_voice(self, area_id: str, channel_id: str) -> None:
        if not self._bot or not self._voice:
            raise RuntimeError("OOPZ voice service is not available")
        join = getattr(self._voice, "join", None)
        if join is None:
            raise RuntimeError("Voice service has no `join` method")
        try:
            result = join(area=area_id, channel=channel_id)
            if inspect.iscoroutine(result):
                result = await result
        except TypeError:
            # try positional
            result = join(area_id, channel_id)
            if inspect.iscoroutine(result):
                result = await result
        self._joined.add((area_id, channel_id))
        logger.info(f"[oopz] joined voice {area_id}/{channel_id}")

    async def leave_voice(self, area_id: str, channel_id: str) -> None:
        if self._voice is not None:
            leave = getattr(self._voice, "leave", None)
            if leave is not None:
                try:
                    result = leave()
                    if inspect.iscoroutine(result):
                        await result
                except Exception as exc:
                    logger.debug(f"[oopz] voice.leave() failed: {exc}")
        self._joined.discard((area_id, channel_id))
        logger.info(f"[oopz] left voice {area_id}/{channel_id}")

    async def push_pcm(
        self,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
        area_id: Optional[str] = None,
        channel_id: Optional[str] = None,
    ) -> None:
        """Push audio bytes to the joined voice channel.

        The OOPZ SDK's `voice.play_bytes` expects a fully-decoded audio blob
        (mp3, wav, ogg...). We wrap the raw PCM in a WAV container so it can
        be played.
        """
        if not self._voice:
            raise RuntimeError("Voice service is not available")
        if not self._joined:
            raise RuntimeError("Bot is not joined to any voice channel")

        # Wrap PCM as WAV so the OOPZ browser backend can play it.
        wav_bytes = _wrap_pcm_as_wav(pcm_bytes, sample_rate, channels, sample_width)

        play_bytes = getattr(self._voice, "play_bytes", None)
        if play_bytes is None:
            raise RuntimeError("Voice service has no `play_bytes` method")

        try:
            result = play_bytes(wav_bytes, mime_type="audio/wav")
        except TypeError:
            # older signature
            result = play_bytes(wav_bytes)
        if inspect.iscoroutine(result):
            await result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _register_decorator(bot: Any, attr: str, handler) -> None:
    """Register a handler via the SDK's `on_<event>` decorator properties."""
    if not hasattr(bot, attr):
        return
    try:
        decorator = getattr(bot, attr)
        if callable(decorator):
            decorated = decorator(handler)
            if decorated is not None and decorated is not handler:
                # Some versions return a registered wrapper, keep it.
                return
    except Exception:
        pass
    # Fallback: a list-style subscriber
    list_attr = f"_{attr}_list"
    if hasattr(bot, list_attr):
        target = getattr(bot, list_attr)
        if isinstance(target, list):
            target.append(handler)


def _safe_get(obj: Any, *names: str) -> Any:
    cur = obj
    for n in names:
        if cur is None:
            return None
        cur = getattr(cur, n, None)
    return cur


def _to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {"value": str(obj)}


def _looks_like_audio_event(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    lowered_keys = {str(k).lower() for k in data.keys()}
    if any(h in k for h in _AUDIO_EVENT_HINTS for k in lowered_keys):
        return True
    return False


def _first(data: Dict[str, Any], keys: tuple, default: Any = None) -> Any:
    for k in keys:
        if k in data:
            return data[k]
    return default


def _extract_audio_frame(data: Dict[str, Any]):
    if isinstance(data, (bytes, bytearray)):
        return "?", "?", "?", bytes(data)
    if not isinstance(data, dict):
        return "?", "?", "?", b""
    area = str(_first(data, _AREA_KEYS, "?"))
    channel = str(_first(data, _CHANNEL_KEYS, "?"))
    user = str(_first(data, _USER_KEYS, "?"))
    pcm: Any = _first(data, _PCM_KEYS, b"")
    if isinstance(pcm, str):
        try:
            pcm = base64.b64decode(pcm)
        except Exception:
            pcm = pcm.encode("utf-8")
    if not isinstance(pcm, (bytes, bytearray)):
        pcm = b""
    return area, channel, user, bytes(pcm)


def _wrap_pcm_as_wav(pcm: bytes, sample_rate: int, channels: int, sample_width: int) -> bytes:
    """Wrap raw PCM into a WAV container (in-memory)."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()
