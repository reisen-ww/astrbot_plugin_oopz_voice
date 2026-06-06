"""
AstrBot plugin entry — OOPZ Voice Chat.

Commands:
    /oopz status                          # list all voice channel states
    /oopz join <area> <channel>           # join a voice channel
    /oopz leave [area] [channel]          # leave (default = all)
    /oopz say <area> <channel> <text>     # direct TTS
    /oopz interrupt [area] [channel]      # interrupt playback
    /oopz set wake <word>                 # change wake word
    /oopz set tts <provider_id>           # switch TTS provider
    /oopz set stt <provider_id>           # switch STT provider
    /oopz set llm <provider_id>           # switch LLM provider
    /oopz history clear [area] [channel]  # clear channel history

WebUI APIs (auto-registered):
    GET  /.../status      — full plugin snapshot
    POST /.../join        — join voice channel
    POST /.../leave       — leave voice channel
    POST /.../interrupt   — interrupt playback
    POST /.../say         — direct TTS
    POST /.../provider    — switch provider
    GET  /.../sse         — Server-Sent Events
    GET  /.../history?area=&channel=  — per-channel conversation
    GET  /.../providers   — list available STT / TTS / LLM providers
    GET  /.../personas    — list available AstrBot personas
"""
from __future__ import annotations

import asyncio
import base64
import inspect
import io
import wave
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

from .conversation_store import ConversationStore
from .vad import VadConfig
from .voice_session import VoiceSession, _ContextHolder


# ---------------------------------------------------------------------------
# SDK import & auto-install
# ---------------------------------------------------------------------------

try:
    from oopz_sdk import OopzBot, OopzConfig  # type: ignore
    _OOPZ_AVAILABLE = True
    _IMPORT_ERROR: Optional[str] = None
except Exception as exc:
    OopzBot = None  # type: ignore
    OopzConfig = None  # type: ignore
    _OOPZ_AVAILABLE = False
    _IMPORT_ERROR = str(exc)


async def _ensure_oopz_sdk_installed(pip_install_timeout: int = 180) -> bool:
    """Install oopz-sdk via pip with --no-deps."""
    global _OOPZ_AVAILABLE, _IMPORT_ERROR, OopzBot, OopzConfig
    if _OOPZ_AVAILABLE:
        return True
    import subprocess
    import sys
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps",
             "--disable-pip-version-check", "oopz-sdk"],
            capture_output=True, text=True, timeout=pip_install_timeout,
        )
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
        logger.warning(f"[oopz] oopz-sdk still not importable: {exc}")
    logger.info(
        "[oopz] oopz-sdk installed but not yet importable. "
        "Please click 'Reload plugin' in the WebUI."
    )
    return False


# ---------------------------------------------------------------------------
# Audio event detection (ported from oopz_client.py)
# ---------------------------------------------------------------------------

_AUDIO_EVENT_HINTS = ("audio", "voice", "frame", "pcm", "opus", "rtc", "track")
_AREA_KEYS = ("area_id", "area", "areaId")
_CHANNEL_KEYS = ("channel_id", "channel", "channelId")
_USER_KEYS = ("user_id", "user", "sender", "uid", "userId")
_PCM_KEYS = ("pcm", "data", "audio", "frame", "samples", "payload")


def _to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {"value": str(obj)}


def _first(data: Dict[str, Any], keys: tuple, default: Any = None) -> Any:
    for k in keys:
        if k in data:
            return data[k]
    return default


def _looks_like_audio_event(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    lowered_keys = {str(k).lower() for k in data.keys()}
    return any(h in k for h in _AUDIO_EVENT_HINTS for k in lowered_keys)


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
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Dashboard routes
# ---------------------------------------------------------------------------

def _register_dashboard_routes(plugin: "OopzVoicePlugin") -> None:
    try:
        from .pages.voice_dashboard.api_handler import register_routes
    except Exception as exc:
        logger.warning(f"[oopz] cannot import dashboard routes: {exc}")
        return
    register_routes(plugin, plugin.context.register_web_api)


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class OopzVoicePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config
        self.context = context
        _ContextHolder.default = context

        self._bot: Any = None
        self._voice: Any = None
        self._connected = False
        self._ready = False
        self._joined_channels: Set[tuple[str, str]] = set()
        self._run_task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

        self._conversation = ConversationStore(
            self,
            max_turns=int((config.get("conversation") or {}).get("max_turns", 12)),
            enable=bool((config.get("conversation") or {}).get("enable_history", True)),
        )
        self._sessions: Dict[str, VoiceSession] = {}
        self._sse_clients: List[asyncio.Queue] = []
        self._sse_lock = asyncio.Lock()
        self._status_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._started = False
        self._terminated = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        _register_dashboard_routes(self)
        if not _OOPZ_AVAILABLE:
            logger.info(
                f"[oopz] oopz-sdk not importable: {_IMPORT_ERROR};"
                " attempting automatic install via pip --no-deps"
            )
            await _ensure_oopz_sdk_installed()
        if not _OOPZ_AVAILABLE:
            logger.warning(f"[oopz] oopz-sdk still unavailable: {_IMPORT_ERROR}")
        await self._try_auto_start()
        logger.info("[oopz] plugin initialized")

    async def terminate(self) -> None:
        self._terminated = True
        self._stopped.set()
        async with self._status_lock:
            sessions = list(self._sessions.values())
        for s in sessions:
            try:
                await s.interrupt()
            except Exception:
                pass
        await self._stop_bot()
        async with self._sse_lock:
            for q in self._sse_clients:
                try:
                    q.put_nowait({"type": "shutdown"})
                except Exception:
                    pass
            self._sse_clients.clear()
        logger.info("[oopz] plugin terminated")

    # ------------------------------------------------------------------
    # SDK bot management
    # ------------------------------------------------------------------

    async def _start_bot(self) -> None:
        if not _OOPZ_AVAILABLE:
            raise RuntimeError(
                f"oopz-sdk is not importable: {_IMPORT_ERROR}. "
                "Install it with `pip install oopz-sdk --no-deps`."
            )
        if self._bot is not None:
            logger.info("[oopz] bot already running")
            return

        auth = self.config.get("auth") or {}
        cfg = OopzConfig(
            device_id=str(auth.get("device_id", "") or "").strip(),
            person_uid=str(auth.get("person_uid", "") or "").strip(),
            jwt_token=str(auth.get("jwt_token", "") or "").strip(),
            private_key=str(auth.get("private_key", "") or ""),
        )
        bot = OopzBot(
            cfg,
            on_ready=self._on_ready,
            on_close=self._on_disconnect,
            on_error=self._on_error,
            on_raw_event=self._on_raw_event,
            on_message=self._on_message,
        )
        self._bot = bot
        self._voice = getattr(bot, "voice", None)
        self._connected = True

        self._run_task = asyncio.create_task(self._run_forever(), name="oopz-bot")
        logger.info("[oopz] bot started")

    async def _stop_bot(self) -> None:
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
        self._voice = None
        self._connected = False
        self._ready = False

    async def _run_forever(self) -> None:
        backoff = 5.0
        while not self._stopped.is_set():
            try:
                run = self._bot.run()
                if inspect.iscoroutine(run):
                    await run
                else:
                    while not self._stopped.is_set() and self._connected:
                        await asyncio.sleep(0.5)
                backoff = 5.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
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

    # ------------------------------------------------------------------
    # SDK event handlers
    # ------------------------------------------------------------------

    async def _on_ready(self, *_args, **_kwargs) -> None:
        self._ready = True
        logger.info("[oopz] ready")
        for area, channel in list(self._joined_channels):
            try:
                await self._voice_join(area, channel)
            except Exception as exc:
                logger.warning(f"[oopz] re-join {area}/{channel} failed: {exc}")

    async def _on_disconnect(self, *_args, **_kwargs) -> None:
        self._ready = False
        logger.warning("[oopz] disconnected")

    async def _on_error(self, *_args, **_kwargs) -> None:
        err = str(_args[0] if _args else _kwargs)
        logger.error(f"[oopz] on_error: {err}")

    async def _on_message(self, message: Any, *_args, **_kwargs) -> None:
        try:
            text = getattr(message, "content", None) or getattr(message, "text", None) or str(message)
            sender = getattr(message, "sender", None)
            sender_name = getattr(sender, "nickname", None) or getattr(sender, "user_id", "?")
            area = getattr(message, "area_id", "?")
            channel = getattr(message, "channel_id", "?")
            logger.debug(f"[oopz] text message {area}/{channel} {sender_name}: {text}")
        except Exception:
            pass

    async def _on_raw_event(self, event: Any) -> None:
        """Sniff for voice frames from the SDK's raw WS event stream."""
        try:
            data = _to_dict(event)
        except Exception:
            data = {"raw": str(event)}
        try:
            if _looks_like_audio_event(data):
                area, channel, user, pcm = _extract_audio_frame(data)
                if pcm:
                    key = f"{area}:{channel}"
                    session = self._sessions.get(key)
                    if session is not None:
                        try:
                            await session.feed_pcm(pcm, user=user)
                        except Exception as exc:
                            logger.warning(f"[oopz] audio feed error {key}: {exc}")
        except Exception as exc:
            logger.debug(f"[oopz] raw event parse error: {exc}")

    # ------------------------------------------------------------------
    # Voice channel control
    # ------------------------------------------------------------------

    async def _voice_join(self, area_id: str, channel_id: str) -> None:
        if not self._voice:
            raise RuntimeError("OOPZ voice service is not available")
        join = getattr(self._voice, "join", None)
        if join is None:
            raise RuntimeError("Voice service has no `join` method")
        try:
            result = join(area=area_id, channel=channel_id)
            if inspect.iscoroutine(result):
                result = await result
        except TypeError:
            result = join(area_id, channel_id)
            if inspect.iscoroutine(result):
                result = await result
        self._joined_channels.add((area_id, channel_id))
        logger.info(f"[oopz] joined voice {area_id}/{channel_id}")

    async def _voice_leave(self, area_id: str, channel_id: str) -> None:
        if self._voice is not None:
            leave = getattr(self._voice, "leave", None)
            if leave is not None:
                try:
                    result = leave()
                    if inspect.iscoroutine(result):
                        await result
                except Exception as exc:
                    logger.debug(f"[oopz] voice.leave() failed: {exc}")
        self._joined_channels.discard((area_id, channel_id))
        logger.info(f"[oopz] left voice {area_id}/{channel_id}")

    async def _voice_play_bytes(self, wav_bytes: bytes) -> None:
        """Push WAV audio to the joined voice channel via SDK."""
        if not self._voice:
            raise RuntimeError("Voice service is not available")
        play_bytes = getattr(self._voice, "play_bytes", None)
        if play_bytes is None:
            raise RuntimeError("Voice service has no `play_bytes` method")
        try:
            result = play_bytes(wav_bytes, mime_type="audio/wav")
        except TypeError:
            result = play_bytes(wav_bytes)
        if inspect.iscoroutine(result):
            await result

    # ------------------------------------------------------------------
    # Auto-start
    # ------------------------------------------------------------------

    async def _try_auto_start(self) -> None:
        async with self._start_lock:
            if self._started or self._terminated:
                return
            auth_ok = all(
                bool((self.config.get("auth") or {}).get(k))
                for k in ("device_id", "person_uid", "jwt_token", "private_key")
            )
            if not auth_ok:
                logger.info("[oopz] auth not fully configured; skipping auto-start")
                return
            try:
                await self._start_bot()
                self._started = True
            except Exception as exc:
                logger.error(f"[oopz] failed to start: {exc}")
                return
            for entry in (self.config.get("auto_join_channels") or []):
                if not isinstance(entry, str) or ":" not in entry:
                    continue
                area, channel = entry.split(":", 1)
                await self._join_channel(area.strip(), channel.strip(), announce=False)

    async def _join_channel(self, area: str, channel: str, announce: bool = True) -> Optional[VoiceSession]:
        if not area or not channel:
            return None
        key = f"{area}:{channel}"
        async with self._status_lock:
            if key in self._sessions:
                return self._sessions[key]
        try:
            await self._voice_join(area, channel)
        except Exception as exc:
            logger.error(f"[oopz] join_voice error: {exc}")
            if announce:
                raise
            return None
        session = self._build_session(area, channel)
        async with self._status_lock:
            self._sessions[key] = session
        if announce:
            await self._broadcast_status({
                "type": "session_joined",
                "key": key,
                "snapshot": session.snapshot.to_dict(),
            })
        return session

    async def _leave_channel(self, area: str, channel: str) -> bool:
        key = f"{area}:{channel}"
        async with self._status_lock:
            session = self._sessions.pop(key, None)
        if session is None:
            return False
        try:
            await session.interrupt()
        except Exception:
            pass
        try:
            await self._voice_leave(area, channel)
        except Exception as exc:
            logger.warning(f"[oopz] leave_voice error: {exc}")
        await self._broadcast_status({"type": "session_left", "key": key})
        return True

    def _build_session(self, area: str, channel: str) -> VoiceSession:
        wake_cfg = self.config.get("wake") or {}
        tts_cfg = self.config.get("tts_playback") or {}
        conv_cfg = self.config.get("conversation") or {}
        whisper_cfg = self.config.get("whisper") or {}
        persona_id = str(conv_cfg.get("persona_id", "") or "").strip()
        custom_sp = str(conv_cfg.get("system_prompt", "") or "")
        system_prompt = _resolve_system_prompt(self.context, persona_id, custom_sp)
        vad_config = VadConfig(
            aggressiveness=int(wake_cfg.get("vad_aggressiveness", 2)),
            rms_gate=float(wake_cfg.get("rms_gate", 0.01)),
            silence_ms_to_flush=int(wake_cfg.get("silence_ms_to_flush", 700)),
            max_listen_ms=int(wake_cfg.get("max_listen_seconds", 30)) * 1000,
            min_listen_ms=int(wake_cfg.get("min_listen_ms", 400)),
            frame_duration_ms=20,
        )
        return VoiceSession(
            area_id=area,
            channel_id=channel,
            on_push_audio=self._voice_play_bytes,
            conversation=self._conversation,
            stt_provider_id=str(self.config.get("stt_provider_id", "") or ""),
            tts_provider_id=str(self.config.get("tts_provider_id", "") or ""),
            llm_provider_id=str(self.config.get("llm_provider_id", "") or ""),
            wake_word=str(wake_cfg.get("wake_word", "bot")),
            wake_variants=list(wake_cfg.get("wake_variants", []) or []),
            whisper_model=str(whisper_cfg.get("model_size", "tiny")),
            whisper_device=str(whisper_cfg.get("device", "cpu")),
            whisper_compute_type=str(whisper_cfg.get("compute_type", "int8")),
            whisper_language=str(whisper_cfg.get("language", "auto")),
            whisper_enabled=bool(whisper_cfg.get("enabled", True)),
            vad_config=vad_config,
            tts_max_text_length=int(tts_cfg.get("max_text_length", 500)),
            tts_split_long_text=bool(tts_cfg.get("split_long_text", True)),
            tts_speed=float(tts_cfg.get("speed", 1.0)),
            system_prompt=system_prompt,
            on_status=self._on_session_status,
            on_log=self._on_session_log,
            on_turn_complete=self._on_turn_complete,
        )

    # ------------------------------------------------------------------
    # Status broadcasting
    # ------------------------------------------------------------------

    async def _on_session_status(self, snapshot: Dict[str, Any]) -> None:
        await self._broadcast_status({"type": "snapshot", "snapshot": snapshot})

    async def _on_session_log(self, role: str, text: str) -> None:
        await self._broadcast_status({"type": "log", "role": role, "text": text})

    async def _on_turn_complete(self, area: str, channel: str, user_text: str, assistant_text: str) -> None:
        try:
            umo = f"oopz_voice:group:{area}:{channel}"
            conv_mgr = self.context.conversation_manager
            cid = await conv_mgr.get_curr_conversation_id(umo)
            if not cid:
                cid = await conv_mgr.new_conversation(umo)
            await conv_mgr.add_message_pair(
                cid,
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            )
        except Exception as exc:
            logger.warning(f"[oopz] conversation_manager push failed: {exc}")

    async def _broadcast_status(self, payload: Dict[str, Any]) -> None:
        async with self._sse_lock:
            dead: List[asyncio.Queue] = []
            for q in self._sse_clients:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    dead.append(q)
                except Exception:
                    dead.append(q)
            for q in dead:
                try:
                    self._sse_clients.remove(q)
                except ValueError:
                    pass

    async def _sse_subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        async with self._sse_lock:
            self._sse_clients.append(q)
        return q

    async def _sse_unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._sse_lock:
            try:
                self._sse_clients.remove(q)
            except ValueError:
                pass

    def _snapshot(self) -> Dict[str, Any]:
        tts_list: List[Dict[str, str]] = []
        stt_list: List[Dict[str, str]] = []
        llm_list: List[Dict[str, str]] = []
        try:
            for p in self.context.get_all_tts_providers() or []:
                pid = getattr(p, "provider_id", "") or getattr(p, "id", "") or ""
                name = getattr(p, "provider_name", "") or getattr(p, "name", "") or pid
                tts_list.append({"id": pid, "name": name})
            for p in self.context.get_all_stt_providers() or []:
                pid = getattr(p, "provider_id", "") or getattr(p, "id", "") or ""
                name = getattr(p, "provider_name", "") or getattr(p, "name", "") or pid
                stt_list.append({"id": pid, "name": name})
            for p in self.context.get_all_providers() or []:
                pid = getattr(p, "provider_id", "") or getattr(p, "id", "") or ""
                name = getattr(p, "provider_name", "") or getattr(p, "name", "") or pid
                llm_list.append({"id": pid, "name": name})
        except Exception as exc:
            logger.debug(f"[oopz] provider enumeration failed: {exc}")
        return {
            "oopz": {
                "connected": self._connected,
                "ready": self._ready,
                "joined": [{"area_id": a, "channel_id": c} for a, c in sorted(self._joined_channels)],
                "sdk_available": _OOPZ_AVAILABLE,
                "sdk_import_error": _IMPORT_ERROR,
            },
            "sessions": [s.snapshot.to_dict() for s in self._sessions.values()],
            "providers": {
                "tts": self.config.get("tts_provider_id"),
                "stt": self.config.get("stt_provider_id"),
                "llm": self.config.get("llm_provider_id"),
                "available_tts": tts_list,
                "available_stt": stt_list,
                "available_llm": llm_list,
            },
            "persona": {
                "active_id": (self.config.get("conversation") or {}).get("persona_id", "") or "",
                "system_prompt": (self.config.get("conversation") or {}).get("system_prompt", "") or "",
            },
            "wake_word": (self.config.get("wake") or {}).get("wake_word", "bot"),
        }

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @filter.command_group("oopz", alias={"oopzvoice"})
    def oopz(self) -> None:
        pass

    @oopz.command("status")
    async def cmd_status(self, event: AstrMessageEvent) -> None:
        snap = self._snapshot()
        sessions = snap["sessions"] or []
        if not sessions:
            yield event.plain_result(
                f"[OOPZ] sdk_available={snap['oopz']['sdk_available']} "
                f"connected={snap['oopz']['connected']} ready={snap['oopz']['ready']}\n"
                "暂无活跃语音会话 / no active voice sessions"
            )
            return
        lines = [f"[OOPZ] connected={snap['oopz']['connected']} ready={snap['oopz']['ready']}"]
        for s in sessions:
            lines.append(
                f"- {s['key']} state={s['state']} turns={s['turn_count']} "
                f"in={s['last_text_in'][:30]!r} out={s['last_text_out'][:30]!r}"
            )
        yield event.plain_result("\n".join(lines))

    @oopz.command("join")
    async def cmd_join(self, event: AstrMessageEvent, area: str, channel: str) -> None:
        if not self._started:
            try:
                await self._start_bot()
                self._started = True
            except Exception as exc:
                yield event.plain_result(f"OOPZ 启动失败: {exc}")
                return
        try:
            session = await self._join_channel(area, channel, announce=False)
        except Exception as exc:
            yield event.plain_result(f"加入失败: {exc}")
            return
        if session is None:
            yield event.plain_result("加入失败：未知原因")
            return
        yield event.plain_result(f"已加入语音频道 {area}:{channel}")

    @oopz.command("leave")
    async def cmd_leave(self, event: AstrMessageEvent, area: Optional[str] = None, channel: Optional[str] = None) -> None:
        if not area and not channel:
            async with self._status_lock:
                keys = list(self._sessions.keys())
            count = 0
            for k in keys:
                a, c = k.split(":", 1)
                if await self._leave_channel(a, c):
                    count += 1
            yield event.plain_result(f"已离开 {count} 个语音频道")
            return
        if not area or not channel:
            yield event.plain_result("用法: /oopz leave <area> <channel>  或  /oopz leave")
            return
        ok = await self._leave_channel(area, channel)
        yield event.plain_result(f"已离开 {area}:{channel}" if ok else f"未在 {area}:{channel} 中")

    @oopz.command("say")
    async def cmd_say(self, event: AstrMessageEvent, area: str, channel: str, text: str) -> None:
        key = f"{area}:{channel}"
        session = self._sessions.get(key)
        if session is None:
            yield event.plain_result(f"未在 {key} 中，请先 /oopz join")
            return
        try:
            await session.say(text)
            yield event.plain_result(f"已在 {key} 播放: {text}")
        except Exception as exc:
            yield event.plain_result(f"播放失败: {exc}")

    @oopz.command("interrupt")
    async def cmd_interrupt(self, event: AstrMessageEvent, area: Optional[str] = None, channel: Optional[str] = None) -> None:
        targets: List[VoiceSession]
        if area and channel:
            s = self._sessions.get(f"{area}:{channel}")
            targets = [s] if s else []
        else:
            targets = [s for s in self._sessions.values() if s.state.value not in ("idle",)]
        for s in targets:
            await s.interrupt()
        yield event.plain_result(f"已打断 {len(targets)} 个会话")

    @oopz.command("set")
    async def cmd_set(self, event: AstrMessageEvent, what: str, value: str) -> None:
        what = what.lower().strip()
        if what == "wake":
            wake_cfg = self.config.get("wake") or {}
            wake_cfg["wake_word"] = value
            self.config["wake"] = wake_cfg
            self.config.save_config()
            for s in self._sessions.values():
                s.update_wake_word(value, list((self.config.get("wake") or {}).get("wake_variants", []) or []))
            yield event.plain_result(f"唤醒词已更新为: {value}")
            return
        if what in ("tts", "stt", "llm"):
            self.config[f"{what}_provider_id"] = value
            self.config.save_config()
            for s in self._sessions.values():
                if what == "tts":
                    s.update_tts_provider(value)
                elif what == "stt":
                    s.update_stt_provider(value)
                else:
                    s.update_llm_provider(value)
            yield event.plain_result(f"{what.upper()} provider 已更新为: {value}")
            return
        yield event.plain_result(f"未知参数: {what}（支持: wake / tts / stt / llm）")

    @oopz.command("history")
    async def cmd_history(self, event: AstrMessageEvent, action: str, area: Optional[str] = None, channel: Optional[str] = None) -> None:
        if action.lower() != "clear":
            yield event.plain_result("用法: /oopz history clear [area] [channel]")
            return
        n = await self._conversation.clear(area, channel)
        if area and channel:
            yield event.plain_result(f"已清空 {area}:{channel} 的历史")
        else:
            yield event.plain_result(f"已清空 {n} 个频道的历史")

    # ------------------------------------------------------------------
    # WebUI API
    # ------------------------------------------------------------------

    async def api_status(self) -> dict:
        return self._snapshot()

    async def api_join(self, area_id: str, channel_id: str) -> dict:
        try:
            await self._join_channel(area_id, channel_id, announce=False)
            return {"ok": True, "snapshot": self._snapshot()}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def api_leave(self, area_id: str, channel_id: str) -> dict:
        ok = await self._leave_channel(area_id, channel_id)
        return {"ok": ok, "snapshot": self._snapshot()}

    async def api_interrupt(self, area_id: str, channel_id: str) -> dict:
        s = self._sessions.get(f"{area_id}:{channel_id}")
        if s is None:
            return {"ok": False, "error": "session not found"}
        await s.interrupt()
        return {"ok": True, "snapshot": s.snapshot.to_dict()}

    async def api_say(self, area_id: str, channel_id: str, text: str) -> dict:
        s = self._sessions.get(f"{area_id}:{channel_id}")
        if s is None:
            return {"ok": False, "error": "session not found"}
        try:
            await s.say(text)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def api_set_provider(self, kind: str, provider_id: str) -> dict:
        kind = kind.lower().strip()
        if kind not in ("tts", "stt", "llm"):
            return {"ok": False, "error": "kind must be tts|stt|llm"}
        self.config[f"{kind}_provider_id"] = provider_id
        self.config.save_config()
        for s in self._sessions.values():
            if kind == "tts":
                s.update_tts_provider(provider_id)
            elif kind == "stt":
                s.update_stt_provider(provider_id)
            else:
                s.update_llm_provider(provider_id)
        return {"ok": True, "snapshot": self._snapshot()}

    async def api_history(self, area_id: str = "", channel_id: str = "") -> dict:
        if area_id and channel_id:
            history = await self._conversation.load(area_id, channel_id)
        else:
            history = []
        return {"ok": True, "history": history, "key": f"{area_id}:{channel_id}"}

    async def api_providers(self) -> dict:
        snap = self._snapshot()
        return {"ok": True, "providers": snap["providers"]}

    async def api_personas(self) -> dict:
        personas: List[Dict[str, str]] = []
        try:
            pm = self.context.persona_manager
            for p in (getattr(pm, "personas_v3", None) or []):
                if isinstance(p, dict):
                    name = p.get("name", "")
                    personas.append({"id": name, "name": name})
            if not personas:
                for p in (getattr(pm, "personas", None) or []):
                    pid = getattr(p, "persona_id", "")
                    name = getattr(p, "persona_name", "") or pid
                    personas.append({"id": pid, "name": name})
        except Exception as exc:
            logger.warning(f"[oopz] persona enumeration failed: {exc}")
        active = (self.config.get("conversation") or {}).get("persona_id", "") or ""
        return {"ok": True, "personas": personas, "active_id": active}

    async def api_sse(self):
        q = await self._sse_subscribe()
        try:
            yield {"event": "snapshot", "data": self._snapshot()}
            while True:
                payload = await q.get()
                if payload.get("type") == "shutdown":
                    break
                yield {"event": "update", "data": payload}
        finally:
            await self._sse_unsubscribe(q)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_system_prompt(context: Context, persona_id: str, custom_prompt: str) -> str:
    if persona_id:
        try:
            pm = getattr(context, "persona_manager", None)
            if pm is not None and hasattr(pm, "get_persona_v3_by_id"):
                persona = pm.get_persona_v3_by_id(persona_id)
                if persona and isinstance(persona, dict):
                    prompt = persona.get("prompt", "")
                    if prompt:
                        return prompt
        except Exception:
            pass
    return custom_prompt
