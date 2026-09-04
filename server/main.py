"""ASL landmark recognition server.

Privacy: accepts landmark geometry only — never video frames.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline.decoder import ContinuousDecoder
from pipeline.gloss_english import gloss_to_english

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("asl.server")

app = FastAPI(
    title="ASL Subtitles Recognition Server",
    description=(
        "Continuous sign-to-text over landmark streams. "
        "Video pixels are never accepted — geometry only."
    ),
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

decoder = ContinuousDecoder()


class LandmarkFrameIn(BaseModel):
    timestamp: float = 0
    hands: list[dict[str, Any]] = Field(default_factory=list)
    body: list[dict[str, Any]] = Field(default_factory=list)
    face: list[dict[str, Any]] = Field(default_factory=list)
    activity: float = 0
    id: str | None = None


class TranslateRequest(BaseModel):
    frames: list[LandmarkFrameIn]
    finalize: bool = True


class TranslateResponse(BaseModel):
    gloss: list[str]
    english: str
    confidence: float
    source: str
    is_final: bool


@app.get("/health")
def health() -> dict[str, Any]:
    payload = {
        "ok": True,
        "privacy": "landmarks-only",
        "model": decoder.model_name,
        "backend": decoder.backend,
        "honesty": "limited-domain continuous; open-domain ASL chat unsolved",
    }
    if getattr(decoder, "uni_sign_meta", None):
        payload["uni_sign"] = decoder.uni_sign_meta
    return payload


@app.post("/v1/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest) -> TranslateResponse:
    frames = [f.model_dump() for f in req.frames]
    result = decoder.decode_window(frames)
    english = gloss_to_english(result.get("gloss") or [])
    return TranslateResponse(
        gloss=result.get("gloss") or [],
        english=english,
        confidence=float(result.get("confidence") or 0),
        source=str(result.get("source") or decoder.backend),
        is_final=req.finalize,
    )


@app.websocket("/v1/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    buffer: list[dict[str, Any]] = []
    max_buf = 96

    await ws.send_json(
        {
            "type": "welcome",
            "connected": True,
            "model": decoder.model_name,
            "message": (
                "Streaming landmarks only. "
                "True open-domain ASL→English is unsolved; this server does limited-domain continuous recognition."
            ),
        }
    )

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "hello":
                await ws.send_json(
                    {
                        "type": "status",
                        "connected": True,
                        "model": decoder.model_name,
                        "message": f"hello-ack protocol={msg.get('protocolVersion', 1)}",
                    }
                )
                continue

            if mtype == "frame":
                frame = msg.get("frame") or msg
                # Strip accidental video fields if a buggy client sends them.
                frame.pop("image", None)
                frame.pop("pixels", None)
                frame.pop("jpeg", None)
                buffer.append(frame)
                if len(buffer) > max_buf:
                    buffer = buffer[-max_buf:]

                # Emit partial every ~8 frames once we have a window.
                if len(buffer) >= 12 and len(buffer) % 8 == 0:
                    result = decoder.decode_window(buffer[-32:])
                    english = gloss_to_english(result.get("gloss") or [])
                    await ws.send_json(
                        {
                            "type": "partial",
                            "gloss": result.get("gloss") or [],
                            "english": english,
                            "confidence": float(result.get("confidence") or 0),
                            "isFinal": False,
                            "source": result.get("source"),
                        }
                    )
                continue

            if mtype == "utterance_end":
                frames = msg.get("frames") or buffer
                for f in frames:
                    if isinstance(f, dict):
                        f.pop("image", None)
                        f.pop("pixels", None)
                result = decoder.decode_window(frames[-64:] if frames else buffer)
                english = gloss_to_english(result.get("gloss") or [])
                await ws.send_json(
                    {
                        "type": "final",
                        "gloss": result.get("gloss") or [],
                        "english": english,
                        "confidence": float(result.get("confidence") or 0),
                        "isFinal": True,
                        "source": result.get("source"),
                    }
                )
                buffer = []
                continue

            if mtype == "ping":
                await ws.send_json({"type": "pong"})
                continue

    except WebSocketDisconnect:
        logger.info("client disconnected")
    except Exception:
        logger.exception("stream error")
        try:
            await ws.close()
        except Exception:
            pass


def run() -> None:
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8765, reload=False)


if __name__ == "__main__":
    run()
