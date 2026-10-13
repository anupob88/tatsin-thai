"""HTTP entrypoint for the Tatsin routing service.

Run:
    python serve.py                                 # 127.0.0.1:8130 (safe default)
    uvicorn app:app --host 127.0.0.1 --port 8130    # same, without the launcher

Endpoints:
    GET  /health           liveness + which models are wired
    GET  /rooms            the room table (id, label, discord channel)
    POST /decide           {state, question?} -> routing decision

Security: every request is authorised by its PEER, not by the bind address (see
`authz.py`). A loopback caller needs no key; anyone else must send `X-Api-Key`
matching `TATSIN_API_KEY`, and if that key is unset a non-loopback request is
refused. The check lives HERE, in the app, because a direct
`uvicorn app:app --host 0.0.0.0` bypasses the bind guard in `serve.py`.

The service answers identity itself, masks PII before anything is stored, and
never asks the user to confirm a routing choice.
"""
from __future__ import annotations

import os
import threading
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from authz import authorize
from clients import JevReaderClient, OpenAIChatClient, TatsinHttpClient
from ladder import Config, Ladder

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("TATSIN_CONFIG", os.path.join(HERE, "config.json"))


class DecideRequest(BaseModel):
    state: str
    question: str | None = None
    source: str = "web"
    debug: bool = False


def build_ladder(cfg: Config) -> Ladder:
    ep = cfg.endpoints
    kind = os.environ.get("TATSIN_SMALL_KIND", "tatsin_http")
    small_ms = cfg.timeouts.get("small_ms", 30000)

    if kind == "openai":
        small = OpenAIChatClient(
            os.environ.get("TATSIN_SMALL_BASE", ""),
            os.environ.get("TATSIN_SMALL_MODEL", "qwen"),
            timeout_ms=small_ms, api_key=os.environ.get("TATSIN_SMALL_KEY"),
            max_tokens=64, style="trained")
    elif kind == "jev_reader":
        small = JevReaderClient(os.environ.get("TATSIN_SMALL_URL", ep.get("jev_reader", "")),
                                timeout_ms=small_ms)
    else:
        small = TatsinHttpClient(
            os.environ.get("TATSIN_SMALL_URL", ep.get("tatsin_decide", "")),
            timeout_ms=small_ms,
            name=os.environ.get("TATSIN_SMALL_NAME", "tatsin"),
            coverage_policy=cfg.coverage_policy)

    big = None
    # Layer 2 is opt-in: set the URL via env (wins) or the config file. An empty
    # value on both means layer 2 is off -- unsure cases go straight to quarantine.
    big_url = os.environ.get("TATSIN_BIG_URL") or ep.get("big_openai", "")
    if big_url:
        big = OpenAIChatClient(
            big_url,
            os.environ.get("TATSIN_BIG_MODEL", ep.get("big_model", "gemma4-12b")),
            timeout_ms=cfg.timeouts.get("big_ms", 60000),
            api_key=os.environ.get("TATSIN_BIG_KEY"),
            max_tokens=96, style="instruct")
    return Ladder(cfg, small, big)


CFG = Config.load(CONFIG_PATH)
LADDER = build_ladder(CFG)
_LOCK = threading.Lock()   # clients share a requests session per call; keep it serialised

app = FastAPI(title="Tatsin router", version="0.1.0")


def _authorize(request: Request, provided: str | None) -> None:
    """Reject a request that may not reach the service.

    The rule keys off the *peer* address, not the bind address, so it holds no
    matter how the server was started -- including a direct
    `uvicorn app:app --host 0.0.0.0`, which bypasses the bind guard in
    `serve.py` entirely. See `authz.py` for the exact rule.
    """
    peer = request.client.host if request.client else None
    rejection = authorize(peer, provided, os.environ.get("TATSIN_API_KEY"))
    if rejection is not None:
        raise HTTPException(status_code=rejection[0], detail=rejection[1])


@app.get("/health")
def health(request: Request,
           x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(request, x_api_key)
    return {
        "ok": True,
        "rooms": len(CFG.rooms),
        "small": getattr(LADDER.small, "model", "?"),
        "big": getattr(LADDER.big, "model", None) if LADDER.big else None,
        "nway_strategy": CFG.nway.get("strategy"),
        "swap_check": CFG.nway.get("swap_check"),
    }


@app.get("/rooms")
def rooms(request: Request,
          x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(request, x_api_key)
    return {"rooms": [{"id": r.id, "label": r.label, "channel_id": r.channel_id}
                      for r in CFG.rooms],
            "quarantine": {"id": CFG.quarantine.id,
                           "label": CFG.quarantine.label,
                           "channel_id": CFG.quarantine.channel_id}}


@app.post("/decide")
def decide(req: DecideRequest, request: Request,
           x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(request, x_api_key)
    if not req.state.strip():
        raise HTTPException(status_code=400, detail="state is empty")
    with _LOCK:
        out = LADDER.decide(req.state, req.question or "ข้อความนี้ควรส่งต่อให้ห้องไหน",
                            source=req.source)
    if not req.debug:
        out.pop("trace", None)
    return out
