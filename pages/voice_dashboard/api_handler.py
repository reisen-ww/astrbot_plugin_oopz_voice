"""
Backend API for the OOPZ Voice Dashboard.

Routes registered on the main plugin instance:
- GET  /.../status       : full snapshot
- POST /.../join         : {area_id, channel_id}
- POST /.../leave        : {area_id, channel_id}
- POST /.../interrupt    : {area_id, channel_id}
- POST /.../say          : {area_id, channel_id, text}
- POST /.../provider     : {kind, provider_id}
- GET  /.../sse          : Server-Sent Events
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from quart import Response, jsonify, request

if TYPE_CHECKING:
    from ..main import OopzVoicePlugin


PLUGIN_NAME = "astrbot_plugin_oopz_voice"


def register_routes(plugin: "OopzVoicePlugin", register_web_api) -> None:
    base = f"/{PLUGIN_NAME}/voice_dashboard"

    async def _status():
        return jsonify(await plugin.api_status())

    async def _join():
        data = await _read_json()
        area = (data.get("area_id") or "").strip()
        channel = (data.get("channel_id") or "").strip()
        if not area or not channel:
            return jsonify({"ok": False, "error": "area_id/channel_id required"}), 400
        return jsonify(await plugin.api_join(area, channel))

    async def _leave():
        data = await _read_json()
        area = (data.get("area_id") or "").strip()
        channel = (data.get("channel_id") or "").strip()
        return jsonify(await plugin.api_leave(area, channel))

    async def _interrupt():
        data = await _read_json()
        area = (data.get("area_id") or "").strip()
        channel = (data.get("channel_id") or "").strip()
        return jsonify(await plugin.api_interrupt(area, channel))

    async def _say():
        data = await _read_json()
        area = (data.get("area_id") or "").strip()
        channel = (data.get("channel_id") or "").strip()
        text = (data.get("text") or "").strip()
        if not area or not channel or not text:
            return jsonify({"ok": False, "error": "area_id/channel_id/text required"}), 400
        return jsonify(await plugin.api_say(area, channel, text))

    async def _provider():
        data = await _read_json()
        kind = (data.get("kind") or "").strip()
        pid = (data.get("provider_id") or "").strip()
        return jsonify(await plugin.api_set_provider(kind, pid))

    async def _sse():
        gen = plugin.api_sse()

        async def stream():
            async for evt in gen:
                event = evt.get("event", "update")
                payload = evt.get("data", {})
                yield f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return Response(stream(), mimetype="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })

    # Provider / Persona / History listing
    async def _providers():
        return jsonify(await plugin.api_providers())

    async def _personas():
        return jsonify(await plugin.api_personas())

    async def _history():
        area = request.args.get("area", "")
        channel = request.args.get("channel", "")
        return jsonify(await plugin.api_history(area, channel))

    # Register routes (register_web_api expects: route, view_handler, methods, desc)
    register_web_api(f"{base}/status", _status, ["GET"], "Voice dashboard: status")
    register_web_api(f"{base}/join", _join, ["POST"], "Voice dashboard: join")
    register_web_api(f"{base}/leave", _leave, ["POST"], "Voice dashboard: leave")
    register_web_api(f"{base}/interrupt", _interrupt, ["POST"], "Voice dashboard: interrupt")
    register_web_api(f"{base}/say", _say, ["POST"], "Voice dashboard: say")
    register_web_api(f"{base}/provider", _provider, ["POST"], "Voice dashboard: switch provider")
    register_web_api(f"{base}/sse", _sse, ["GET"], "Voice dashboard: SSE")
    register_web_api(f"{base}/providers", _providers, ["GET"], "Voice dashboard: list providers")
    register_web_api(f"{base}/personas", _personas, ["GET"], "Voice dashboard: list personas")
    register_web_api(f"{base}/history", _history, ["GET"], "Voice dashboard: conversation history")


async def _read_json() -> dict:
    try:
        return await request.get_json(force=True, silent=True) or {}
    except Exception:
        return {}