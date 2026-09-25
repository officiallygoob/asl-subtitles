#!/usr/bin/env python3
"""LAN LiveKit token server for ASL Subtitles 1:1 calls.

Identity is a username plus a room code. This process never receives video or audio.
Do not expose it to the public internet — anyone who can reach it can mint a room token.

Run (LiveKit Cloud free tier or your own LiveKit server):

    cd server/livekit
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    export LIVEKIT_URL=wss://YOUR_PROJECT.livekit.cloud
    export LIVEKIT_API_KEY=...
    export LIVEKIT_API_SECRET=...
    python token_server.py

Then point each phone's Call tab at http://<this-mac-lan-ip>:8787/v1/token.
"""

from __future__ import annotations

import json
import os
import re
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from username_policy import validate_username

ROOM_RE = re.compile(r"^[a-z0-9]{4,64}$")
MAX_BODY = 8192


def normalize_room(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    room = raw.strip().lower()
    if ROOM_RE.fullmatch(room) is None:
        return None
    return room


def mint_token(room: str, username: str) -> dict:
    url = os.environ.get("LIVEKIT_URL", "").strip()
    api_key = os.environ.get("LIVEKIT_API_KEY", "").strip()
    api_secret = os.environ.get("LIVEKIT_API_SECRET", "").strip()
    if not url or not api_key or not api_secret:
        raise RuntimeError(
            "Token server is missing LIVEKIT_URL, LIVEKIT_API_KEY, or LIVEKIT_API_SECRET."
        )
    try:
        from livekit import api
    except ImportError as exc:
        raise RuntimeError(
            "livekit-api is not installed. From server/livekit: pip install -r requirements.txt"
        ) from exc

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(username)
        .with_name(username)
        .with_ttl(timedelta(hours=6))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )
    return {"url": url, "token": token, "room": room, "identity": username}


class TokenHandler(BaseHTTPRequestHandler):
    server_version = "ASLCallToken/1.0"

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/health":
            configured = bool(
                os.environ.get("LIVEKIT_URL")
                and os.environ.get("LIVEKIT_API_KEY")
                and os.environ.get("LIVEKIT_API_SECRET")
            )
            self._send(200, {"ok": True, "livekitConfigured": configured})
            return
        self._send(404, {"error": "Not found."})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/v1/token":
            self._send(404, {"error": "Not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, {"error": "Invalid Content-Length."})
            return
        if length < 0 or length > MAX_BODY:
            self._send(413, {"error": "Request too large."})
            return
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send(400, {"error": "Body must be JSON."})
            return
        if not isinstance(body, dict):
            self._send(400, {"error": "Body must be a JSON object."})
            return

        room = normalize_room(body.get("room"))
        if room is None:
            self._send(400, {"error": "Room code must be 4–64 letters or digits."})
            return
        status, name_or_reason = validate_username(str(body.get("username", "")))
        if status != "ok":
            self._send(400, {"error": name_or_reason})
            return

        try:
            payload = mint_token(room, name_or_reason)
        except RuntimeError as exc:
            self._send(500, {"error": str(exc)})
            return
        # Log the room and username only — never the JWT or API secret.
        print(f"minted token room={room} identity={name_or_reason}", flush=True)
        self._send(200, payload)

    def log_message(self, fmt: str, *args) -> None:
        # Default logging can include request lines; keep it, but mint_token does not log secrets.
        super().log_message(fmt, *args)

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    port = int(os.environ.get("PORT", "8787"))
    httpd = ThreadingHTTPServer(("0.0.0.0", port), TokenHandler)
    print(f"ASL call token server on 0.0.0.0:{port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
