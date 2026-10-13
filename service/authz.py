"""Request authorisation for the Tatsin routing service.

Authorisation keys off the *peer*, not the bind address. A request that arrives
over loopback is local, so it needs no key. A request from any other address
MUST carry `TATSIN_API_KEY`; if no key is configured the request is refused
(fail closed). This is enforced *inside the app*, so it holds no matter how the
server was started -- `serve.py` (which also refuses a non-loopback bind up
front) or `uvicorn app:app --host ...` directly (which does not check the bind).

The rule is deliberately about `request.client.host`: the bind address is a
start-up choice that a direct `uvicorn` launch can set to `0.0.0.0`, so it can
never be the thing that decides whether a caller is trusted.
"""
from __future__ import annotations

import ipaddress
from typing import Optional

# (status_code, detail) to reject a request, or None to allow it.
Rejection = Optional[tuple[int, str]]


def is_loopback(host: str | None) -> bool:
    """True only for an address that can only be the local machine.

    An empty or unparseable host is NOT loopback: we cannot prove it is local,
    so it is treated as remote and made to authenticate.
    """
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def authorize(peer_host: str | None, provided_key: str | None,
              expected_key: str | None) -> Rejection:
    """Decide whether a request may proceed. Returns None to allow."""
    if is_loopback(peer_host):
        return None
    if not expected_key:
        return (403, "non-loopback request refused: TATSIN_API_KEY is not set, "
                     "so this request cannot be authenticated")
    if provided_key != expected_key:
        return (401, "bad or missing api key")
    return None
