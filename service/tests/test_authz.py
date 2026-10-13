"""Tests for peer-based authorisation.

Runs with the Python standard library plus FastAPI/Starlette (already required
to run the service itself); no model weights and no network:

    cd service
    python -m unittest discover -s tests -v

Two layers are covered:

  * `authz.is_loopback` / `authz.authorize` -- the rule itself, in isolation.
  * the app wiring -- that EVERY endpoint calls the check, so a request that
    arrives directly (`uvicorn app:app --host 0.0.0.0`) is refused when it comes
    from a non-loopback peer and no key is configured. This is the hole the
    tests exist to keep shut: the check must live in the app, not only in the
    launcher, because a direct launch bypasses the launcher entirely.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from starlette.requests import Request  # noqa: E402

import app as app_module  # noqa: E402
from authz import authorize, is_loopback  # noqa: E402


def make_request(host: str | None) -> Request:
    """A minimal ASGI scope carrying just what `_authorize` reads: the peer."""
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [],
        "client": (host, 12345) if host is not None else None,
        "server": ("127.0.0.1", 8130),
        "scheme": "http",
    }
    return Request(scope)


class IsLoopback(unittest.TestCase):
    def test_loopback_addresses(self):
        for host in ("127.0.0.1", "127.0.0.5", "::1", "localhost"):
            self.assertTrue(is_loopback(host), host)

    def test_empty_host_is_not_loopback(self):
        # uvicorn binds "" to every interface, so "" must NOT count as local.
        self.assertFalse(is_loopback(""))

    def test_wildcard_bind_is_not_loopback(self):
        self.assertFalse(is_loopback("0.0.0.0"))
        self.assertFalse(is_loopback("::"))

    def test_lan_and_public_are_not_loopback(self):
        for host in ("192.168.1.5", "10.0.0.7", "203.0.113.7"):
            self.assertFalse(is_loopback(host), host)

    def test_unparseable_and_missing_are_not_loopback(self):
        # Cannot prove it is local -> treat as remote and demand a key.
        for host in ("not-an-ip", "example.com", None):
            self.assertFalse(is_loopback(host), host)


class Authorize(unittest.TestCase):
    def test_loopback_needs_no_key(self):
        self.assertIsNone(authorize("127.0.0.1", None, None))
        self.assertIsNone(authorize("127.0.0.1", None, "secret"))

    def test_remote_with_no_key_configured_is_refused(self):
        # Fail closed: no key anywhere means a remote caller cannot be trusted.
        status, _ = authorize("203.0.113.7", None, None)
        self.assertEqual(status, 403)

    def test_remote_with_missing_key_is_rejected(self):
        status, _ = authorize("203.0.113.7", None, "secret")
        self.assertEqual(status, 401)

    def test_remote_with_wrong_key_is_rejected(self):
        status, _ = authorize("203.0.113.7", "guess", "secret")
        self.assertEqual(status, 401)

    def test_remote_with_correct_key_is_allowed(self):
        self.assertIsNone(authorize("203.0.113.7", "secret", "secret"))


class AppEnforcement(unittest.TestCase):
    """Every endpoint must refuse a remote, unauthenticated caller.

    This is what makes the fix real: the guard is exercised through the endpoint
    functions, so deleting the `_authorize` call from any one of them turns a
    test red.
    """

    def setUp(self):
        self._saved = os.environ.pop("TATSIN_API_KEY", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("TATSIN_API_KEY", None)
        else:
            os.environ["TATSIN_API_KEY"] = self._saved

    def _assert_403(self, call):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            call()
        self.assertEqual(ctx.exception.status_code, 403)

    def test_health_refuses_remote_without_key(self):
        self._assert_403(lambda: app_module.health(make_request("203.0.113.7"), None))

    def test_rooms_refuses_remote_without_key(self):
        self._assert_403(lambda: app_module.rooms(make_request("203.0.113.7"), None))

    def test_decide_refuses_remote_without_key(self):
        req = app_module.DecideRequest(state="งานอะไรสักอย่าง")
        self._assert_403(
            lambda: app_module.decide(req, make_request("203.0.113.7"), None))

    def test_direct_uvicorn_launch_from_remote_is_not_a_bypass(self):
        # The scenario Tonhop flagged: someone runs `uvicorn app:app --host
        # 0.0.0.0` directly, skipping serve.py. A remote request still hits the
        # in-app check and is refused, because the check reads the PEER.
        req = app_module.DecideRequest(state="งานอะไรสักอย่าง")
        self._assert_403(
            lambda: app_module.decide(req, make_request("198.51.100.9"), None))

    def test_loopback_is_allowed_through_the_check(self):
        # Assert the guard PASSES for a local caller without running the ladder.
        self.assertIsNone(app_module._authorize(make_request("127.0.0.1"), None))

    def test_remote_with_correct_key_passes_the_check(self):
        os.environ["TATSIN_API_KEY"] = "secret"
        self.assertIsNone(
            app_module._authorize(make_request("203.0.113.7"), "secret"))

    def test_remote_with_wrong_key_is_401_not_403(self):
        from fastapi import HTTPException
        os.environ["TATSIN_API_KEY"] = "secret"
        with self.assertRaises(HTTPException) as ctx:
            app_module._authorize(make_request("203.0.113.7"), "guess")
        self.assertEqual(ctx.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
