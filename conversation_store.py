"""
Per-channel conversation history, persisted in AstrBot's KV store.

Each OOPZ voice channel is keyed by `area_id:channel_id`. We keep a sliding
window of `max_turns` user/assistant turns and prepend the system prompt at
LLM-call time.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from astrbot.api import logger


def channel_key(area_id: str, channel_id: str) -> str:
    return f"{area_id}:{channel_id}"


class ConversationStore:
    def __init__(self, star_instance, max_turns: int = 12, enable: bool = True) -> None:
        self._star = star_instance
        self._max_turns = max(1, int(max_turns))
        self._enable = enable
        self._mem_cache: Dict[str, List[dict]] = {}
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._enable

    def set_enabled(self, value: bool) -> None:
        self._enable = bool(value)

    def set_max_turns(self, max_turns: int) -> None:
        self._max_turns = max(1, int(max_turns))

    async def load(self, area_id: str, channel_id: str) -> List[dict]:
        if not self._enable:
            return []
        key = self._kv_key(area_id, channel_id)
        async with self._lock:
            if key in self._mem_cache:
                return list(self._mem_cache[key])
        try:
            data = await self._star.get_kv_data(key, [])
        except Exception as exc:
            logger.warning(f"[oopz] kv load error for {key}: {exc}")
            data = []
        if not isinstance(data, list):
            data = []
        async with self._lock:
            self._mem_cache[key] = data
        return list(data)

    async def append(self, area_id: str, channel_id: str, role: str, content: str) -> None:
        if not self._enable:
            return
        if role not in {"user", "assistant"}:
            return
        content = (content or "").strip()
        if not content:
            return
        key = self._kv_key(area_id, channel_id)
        history = await self.load(area_id, channel_id)
        history.append({"role": role, "content": content})
        # Trim
        if len(history) > self._max_turns * 2:
            history = history[-self._max_turns * 2:]
        async with self._lock:
            self._mem_cache[key] = history
        try:
            await self._star.put_kv_data(key, history)
        except Exception as exc:
            logger.warning(f"[oopz] kv save error for {key}: {exc}")

    async def clear(self, area_id: Optional[str] = None, channel_id: Optional[str] = None) -> int:
        """Clear one channel's history (or all if both are None). Returns count cleared."""
        cleared = 0
        async with self._lock:
            keys = list(self._mem_cache.keys())
        if area_id is None and channel_id is None:
            for k in keys:
                await self._delete_key(k)
                cleared += 1
            return cleared
        if area_id is None or channel_id is None:
            return 0
        k = self._kv_key(area_id, channel_id)
        await self._delete_key(k)
        return 1

    async def _delete_key(self, key: str) -> None:
        async with self._lock:
            self._mem_cache.pop(key, None)
        try:
            await self._star.delete_kv_data(key)
        except Exception as exc:
            logger.warning(f"[oopz] kv delete error for {key}: {exc}")

    def _kv_key(self, area_id: str, channel_id: str) -> str:
        return f"oopz_voice:history:{channel_key(area_id, channel_id)}"
