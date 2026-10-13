"""POST /tatsin/decide -- layer-1 (small model) server for the Tatsin router.

Serves the project's fine-tuned checkpoint through the RLAgent wrapper and
answers the contract the wrapper expects (see ../README.md section 6):

    POST {state, question, options[]}
     -> {pick, confidence, noul, margin, ranked, coverage{scored,requested}, source}

Ported from the trainer's rig so it installs anywhere:
  * paths are relative + env-driven (no hard-coded absolute paths),
  * it binds 127.0.0.1 by default and refuses a non-loopback bind unless
    ``TATSIN_API_KEY`` is set -- an unauthenticated routing endpoint on the LAN
    is an open relay into your models,
  * the checkpoint state dict is read with safetensors' ``pread`` backend,
    which loads the same file with far less commit headroom than the default
    ``mmap`` (see the note on ``_use_pread_backend``).

Env:
    TATSIN_CKPT                 checkpoint dir         (default: ./ckpt/tatsin_v2)
    TATSIN_SOURCE_TAG           the `source` string    (default: ckpt dir's basename)
    TATSIN_HOST / TATSIN_PORT   bind                   (default 127.0.0.1:8140)
    TATSIN_API_KEY              required for a non-loopback bind; checked on POST
    TATSIN_DEVICE               cpu | cuda             (default: cpu)
    TATSIN_MIN_COMMIT_GB        abort below this       (default 2.5, see note)
    TATSIN_SAFETENSORS_BACKEND  pread | mmap            (default: pread)
"""
import argparse
import ipaddress
import json
import os
import sys
import threading
import time

os.environ.setdefault("USE_TF", "0")  # transformers probes for TF at import; its abseil runtime can deadlock model build run with TF off

HERE = os.path.dirname(os.path.abspath(__file__))
# The RLAgent API ships next to this file (`rl_agent_api.py`, `rl_common.py`).
sys.path.insert(0, os.environ.get("TATSIN_RLAPI_DIR", HERE))

CKPT_DIR = os.environ.get("TATSIN_CKPT", os.path.join(HERE, "ckpt", "tatsin_v2"))
DEVICE = os.environ.get("TATSIN_DEVICE", "cpu")
API_KEY = os.environ.get("TATSIN_API_KEY")
MIN_COMMIT_GB = float(os.environ.get("TATSIN_MIN_COMMIT_GB", "2.5"))
SAFETENSORS_BACKEND = os.environ.get("TATSIN_SAFETENSORS_BACKEND", "pread")

# `frame` is MODEL-BOUND: the checkpoint was trained and QC'd with the state
# wrapped in exactly this template, so the endpoint applies it and the model
# always sees the distribution it knows. Do not "tidy" the wording.
FRAME = 'พ่อส่งข้อความเข้ามา: "{inner}"'
NOUL_CRIT = {
    "false": "เป็นงานที่ต้องส่งต่อให้ห้องเฉพาะทางจัดการ",
    "true": "ไม่ใช่การส่งต่อห้อง — เป็นคำทักทาย/คำถามทั่วไป/นอกขอบเขต ตอบเองได้",
}
ROUTE_INS = "ข้อความนี้ควรส่งต่อให้ห้องไหน"
LETTERS = "ABCDEFGH"
# The `source` string the wrapper matches against gates.noul_is_exact_by_model.
# Default = the checkpoint dir's own name, so naming the folder after the
# checkpoint is enough; TATSIN_SOURCE_TAG overrides it for other layouts.
SOURCE_TAG = os.environ.get("TATSIN_SOURCE_TAG", os.path.basename(CKPT_DIR.rstrip("/\\")))


def commit_headroom_gb():
    """ullAvailPageFile ~= CommitLimit - CommittedBytes (the tightest limit on Windows)."""
    try:
        import ctypes

        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        ms = MemStatus()
        ms.dwLength = ctypes.sizeof(MemStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return None
        return round(ms.ullAvailPageFile / (1024 ** 3), 2)
    except Exception:
        return None


def _use_pread_backend(backend: str) -> None:
    """Default the state-dict read to ``pread`` instead of safetensors' ``mmap``.

    ``safetensors.torch.load_file`` defaults to ``backend="mmap"``, which maps
    the whole checkpoint and then faults with a Windows access violation once
    system commit is tight (measured: a 1 KB tensor read crashed at ~1.8 GB free
    commit while a resident process held ~15 GB). The ``pread`` path reads the
    tensor bytes directly and loaded the same 1.29 GB state dict with far less
    headroom. mmap is fine on roomy hosts; set TATSIN_SAFETENSORS_BACKEND=mmap
    to opt back in.
    """
    if backend == "mmap":
        return
    import safetensors.torch as st
    from safetensors import safe_open

    def load_file(filename, device="cpu", **_kw):
        with safe_open(filename, framework="pt", device=device, backend=backend) as f:
            return f.get_tensors()

    st.load_file = load_file


HEADROOM = commit_headroom_gb()
if HEADROOM is not None and HEADROOM < MIN_COMMIT_GB:
    print(f"FATAL: system commit headroom {HEADROOM} GB < TATSIN_MIN_COMMIT_GB "
          f"({MIN_COMMIT_GB}). Free memory (or lower TATSIN_MIN_COMMIT_GB if you "
          "know the box can take it), then retry.", flush=True)
    sys.exit(1)

_use_pread_backend(SAFETENSORS_BACKEND)
from rl_agent_api import RLAgent  # noqa: E402  (after the headroom guard + loader patch)

_T0 = time.perf_counter()
agent = RLAgent(CKPT_DIR, device=DEVICE)
LOAD_S = time.perf_counter() - _T0

_LOCK = threading.Lock()  # one forward pass at a time


def decide(state_raw: str, question, options: list[str]) -> dict:
    ins = question if isinstance(question, str) and question.strip() else ROUTE_INS
    crit = {LETTERS[i]: opt for i, opt in enumerate(options)}
    state = FRAME.format(inner=state_raw)
    # Both heads asked per call: the route choice + the scope noul. The 2-layer
    # gate (noul, margin) travels back to the wrapper; the gate itself lives there.
    qs = {
        "route": {"type": "choice", "instructions": ins, "criteria": crit},
        "scope": {"type": "noul", "instructions": ins, "criteria": dict(NOUL_CRIT)},
    }
    with _LOCK:
        t0 = time.perf_counter()
        out = agent.system_one(state, qs)
        wall_ms = (time.perf_counter() - t0) * 1000
    route, scope = out["answers"]["route"], out["answers"]["scope"]
    probs = dict(route["probabilities"])
    top = sorted(probs.values(), reverse=True)
    margin = round(top[0] - top[1], 4) if len(top) >= 2 else round(top[0], 4)
    return {
        "source": SOURCE_TAG,
        "pick": route["choice"],
        "confidence": route["confidence"],
        "noul": scope["noul"],
        "margin": margin,
        "ranked": probs,
        # coverage is mandatory in the contract: full by construction, or the
        # call fails loudly (422) -- never a silent partial slot.
        "coverage": {"scored": len(probs), "requested": len(options)},
        "latency_ms": round(wall_ms, 1),
    }


from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _key_ok(self) -> bool:
        return API_KEY is None or self.headers.get("X-Api-Key") == API_KEY

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "source": SOURCE_TAG,
                             "ckpt": os.path.basename(CKPT_DIR), "device": DEVICE,
                             "load_s": round(LOAD_S, 1),
                             "commit_headroom_gb_now": commit_headroom_gb()})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/tatsin/decide":
            self._send(404, {"error": "not found"})
            return
        if not self._key_ok():
            self._send(401, {"error": "bad or missing api key"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 1_000_000:
                self._send(413, {"error": "body too large"})
                return
            req = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._send(400, {"error": f"unreadable body: {type(exc).__name__}: {exc}"})
            return
        state = req.get("state")
        options = req.get("options")
        if not isinstance(state, str) or not state.strip():
            self._send(400, {"error": "state must be a non-empty string"})
            return
        if (not isinstance(options, list) or not options
                or not all(isinstance(o, str) and o.strip() for o in options)):
            self._send(400, {"error": "options must be a non-empty array of non-empty strings"})
            return
        if len(options) > len(LETTERS):
            self._send(400, {"error": f"max {len(LETTERS)} options (A-H)"})
            return
        try:
            self._send(200, decide(state, req.get("question"), options))
        except ValueError as exc:
            # e.g. options do not fit in head_max_len -- all-or-error, never silent
            self._send(422, {"error": str(exc),
                             "coverage": {"scored": 0, "requested": len(options)}})
        except Exception as exc:
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt, *args):  # keep the console useful
        print(f"[tatsin-decide] {self.address_string()} {fmt % args}", flush=True)


def _is_loopback(host: str) -> bool:
    # An empty host is NOT loopback: the server binds "" to every interface, so
    # `TATSIN_HOST=` must fail the guard below rather than slip past it.
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> int:
    p = argparse.ArgumentParser(description="Run the Tatsin layer-1 (small model) server.")
    p.add_argument("--host", default=os.environ.get("TATSIN_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("TATSIN_PORT", "8140")))
    args = p.parse_args()

    if not _is_loopback(args.host) and not API_KEY:
        print(
            f"refusing to bind {args.host}:{args.port} without TATSIN_API_KEY --\n"
            "that exposes an UNAUTHENTICATED model endpoint to the network.\n"
            "Set TATSIN_API_KEY (requests must then send X-Api-Key), or bind 127.0.0.1.",
            file=sys.stderr,
        )
        return 2

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[tatsin-decide] listening {args.host}:{args.port} source={SOURCE_TAG} "
          f"ckpt={CKPT_DIR} device={DEVICE} load={LOAD_S:.1f}s "
          f"headroom={HEADROOM}GB backend={SAFETENSORS_BACKEND}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
