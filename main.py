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
    GET  /.../sse         — Server‑Sent Events
    GET  /.../history?area=&channel=  — per‑channel conversation
    GET  /.../providers   — list available STT / TTS / LLM providers
    GET  /.../personas    — list available AstrBot personas
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

from .conversation_store import ConversationStore
from .oopz_client import OopzAuth, OopzClient, ensure_oopz_sdk_installed
from .oopz_event_router import OopzEventRouter
from .vad import VadConfig
from .voice_session import VoiceSession, _ContextHolder


def _register_dashboard_routes(plugin: "OopzVoicePlugin") -> None:
    """Register WebUI routes against the plugin's context.

    Imported lazily so a missing `pages/` folder doesn't break the plugin.
    """
    try:
        from .pages.voice_dashboard.api_handler import register_routes
    except Exception as exc:  # pragma: no cover - filesystem race
        logger.warning(f"[oopz] cannot import dashboard routes: {exc}")
        return
    register_routes(plugin, plugin.context.register_web_api)


class OopzVoicePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config
        self.context = context
        _ContextHolder.default = context

        self._oopz = OopzClient(_read_auth(config))
        self._router = OopzEventRouter(self._oopz)
        self._conversation = ConversationStore(
            self,
            max_turns=int((config.get("conversation") or {}).get("max_turns", 12)),
            enable=bool((config.get("conversation") or {}).get("enable_history", True)),
        )
        self._sessions: Dict[str, VoiceSession] = {}
        self._status_listeners: List[Callable[[Dict[str, Any]], Awaitable[None]]] = []
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
        await self._router.start()
        _register_dashboard_routes(self)
        if not self._oopz.status.sdk_available:
            logger.info(
                f"[oopz] oopz-sdk not importable: {self._oopz.status.sdk_import_error};"
                " attempting automatic install via pip --no-deps"
            )
            await ensure_oopz_sdk_installed()
        if not self._oopz.status.sdk_available:
            logger.warning(
                f"[oopz] oopz-sdk still unavailable: {self._oopz.status.sdk_import_error}"
            )
        await self._try_auto_start()
        logger.info("[oopz] plugin initialized")

    async def terminate(self) -> None:
        self._terminated = True
        async with self._status_lock:
            sessions = list(self._sessions.values())
        for s in sessions:
            try:
                await s.interrupt()
            except Exception:
                pass
        try:
            await self._oopz.stop()
        except Exception as exc:
            logger.warning(f"[oopz] stop error: {exc}")
        async with self._sse_lock:
            for q in self._sse_clients:
                try:
                    q.put_nowait({"type": "shutdown"})
                except Exception:
                    pass
            self._sse_clients.clear()
        logger.info("[oopz] plugin terminated")

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
                await self._oopz.start()
                self._started = True
            except Exception as exc:
                logger.error(f"[oopz] failed to start: {exc}")
                return
            for entry in (self.config.get("auto_join_channels") or []):
                if not isinstance(entry, str):
                    continue
                if ":" not in entry:
                    logger.warning(f"[oopz] bad auto_join entry: {entry!r}")
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
            await self._oopz.join_voice(area, channel)
        except Exception as exc:
            logger.error(f"[oopz] join_voice error: {exc}")
            if announce:
                raise
            return None
        session = self._build_session(area, channel)
        async with self._status_lock:
            self._sessions[key] = session
        await self._router.register(session)
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
        await self._router.unregister(area, channel)
        try:
            await session.interrupt()
        except Exception:
            pass
        try:
            await self._oopz.leave_voice(area, channel)
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
            oopz=self._oopz,
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
        """Push a completed voice turn into AstrBot's conversation manager."""
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
        # Providers — list all registered by type
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
                "connected": self._oopz.status.connected,
                "ready": self._oopz.status.ready,
                "joined": [c.__dict__ for c in self._oopz.status.joined],
                "last_error": self._oopz.status.last_error,
                "sdk_available": self._oopz.status.sdk_available,
                "sdk_import_error": self._oopz.status.sdk_import_error,
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
                await self._oopz.start()
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
        """Return per-channel conversation history."""
        if area_id and channel_id:
            history = await self._conversation.load(area_id, channel_id)
        else:
            history = []
        return {"ok": True, "history": history, "key": f"{area_id}:{channel_id}"}

    async def api_providers(self) -> dict:
        """Return available STT/TTS/LLM providers from AstrBot."""
        snap = self._snapshot()
        return {"ok": True, "providers": snap["providers"]}

    async def api_personas(self) -> dict:
        """Return available AstrBot personas (sync in-memory)."""
        personas: List[Dict[str, str]] = []
        try:
            pm = self.context.persona_manager
            for p in (getattr(pm, "personas_v3", None) or []):
                if isinstance(p, dict):
                    name = p.get("name", "")
                    pid = name
                    personas.append({"id": pid, "name": name})
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
        """Async generator for Server-Sent Events."""
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
    """Return the effective system prompt: persona prompt > custom > default.

    Uses `persona_manager.get_persona_v3_by_id()` (sync, in-memory).
    """
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


def _read_auth(config: AstrBotConfig) -> OopzAuth:
    auth = config.get("auth") or {}
    return OopzAuth(
        device_id=str(auth.get("device_id", "") or "").strip(),
        person_uid=str(auth.get("person_uid", "") or "").strip(),
        jwt_token=str(auth.get("jwt_token", "") or "").strip(),
        private_key=str(auth.get("private_key", "") or ""),
    )
