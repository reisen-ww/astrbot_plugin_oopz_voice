"""
Routes audio frames coming from the OOPZ client to the right VoiceSession.
"""
from __future__ import annotations

import asyncio
from typing import Dict, Optional

from astrbot.api import logger

from .oopz_client import OopzClient
from .voice_session import VoiceSession


class OopzEventRouter:
    def __init__(self, oopz: OopzClient) -> None:
        self._oopz = oopz
        self._sessions: Dict[str, VoiceSession] = {}
        self._lock = asyncio.Lock()
        self._subscribed = False

    async def start(self) -> None:
        if self._subscribed:
            return
        self._oopz.subscribe_audio(self._on_audio)
        self._oopz.subscribe_voice_event(self._on_voice_event)
        self._subscribed = True
        logger.info("[oopz] event router started")

    async def register(self, session: VoiceSession) -> None:
        key = f"{session.area_id}:{session.channel_id}"
        async with self._lock:
            self._sessions[key] = session

    async def unregister(self, area_id: str, channel_id: str) -> None:
        key = f"{area_id}:{channel_id}"
        async with self._lock:
            self._sessions.pop(key, None)

    def get(self, area_id: str, channel_id: str) -> Optional[VoiceSession]:
        return self._sessions.get(f"{area_id}:{channel_id}")

    def all(self) -> Dict[str, VoiceSession]:
        return dict(self._sessions)

    async def _on_audio(self, area: str, channel: str, user: str, pcm: bytes) -> None:
        key = f"{area}:{channel}"
        session = self._sessions.get(key)
        if session is None:
            return
        try:
            await session.feed_pcm(pcm, user=user)
        except Exception as exc:
            logger.warning(f"[oopz] audio feed error {key}: {exc}")

    async def _on_voice_event(self, event: dict) -> None:
        # Logged at DEBUG; we don't act on it for now.
        logger.debug(f"[oopz] voice event: {event}")
