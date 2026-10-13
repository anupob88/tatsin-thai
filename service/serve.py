"""Safe launcher for the Tatsin routing service.

Binds to LOOPBACK by default (127.0.0.1) so a fresh install is not reachable
from the local network. Binding anywhere else turns the endpoint into a network
service, which is refused unless an access key (`TATSIN_API_KEY`) is set --
otherwise anyone who can reach the port could push arbitrary text through your
models and read the routed result, with no authentication at all.

    python serve.py                                   # 127.0.0.1:8130
    python serve.py --host 0.0.0.0                    # refused (no key)
    TATSIN_API_KEY=... python serve.py --host 0.0.0.0 # opt in, auth required

Env: TATSIN_HOST / TATSIN_PORT / TATSIN_API_KEY override the defaults;
`--host` / `--port` flags win over the env.
"""
from __future__ import annotations

import argparse
import os
import sys

import uvicorn

from authz import is_loopback


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the Tatsin routing service.")
    p.add_argument("--host", default=os.environ.get("TATSIN_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("TATSIN_PORT", "8130")))
    args = p.parse_args(argv)

    # `is_loopback` comes from authz so the launcher and the app agree on what
    # "local" means. Notably "" is NOT loopback: uvicorn binds an empty host to
    # every interface, so treating it as local would let `TATSIN_HOST=` slip past
    # this guard and expose the endpoint -- the app-level check would still stop
    # it, but the launcher should refuse up front rather than rely on that.
    if not is_loopback(args.host) and not os.environ.get("TATSIN_API_KEY"):
        print(
            f"refusing to bind {args.host}:{args.port} without TATSIN_API_KEY --\n"
            "that exposes an UNAUTHENTICATED routing endpoint to the network.\n"
            "Set TATSIN_API_KEY (requests must then send X-Api-Key), or bind 127.0.0.1.",
            file=sys.stderr,
        )
        return 2

    uvicorn.run("app:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
