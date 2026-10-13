"""Tests for the model clients and the pre-model PDPA mask.

    cd service
    python -m unittest discover -s tests -v

The HTTP client is exercised against a fake `requests.post`, so the endpoint
contract (coverage, noul, pick) is pinned without a live server.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients import (OpenAIChatClient, TatsinHttpClient, letter,  # noqa: E402
                     mask_pii)


class _Resp:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _decide(body: dict):
    client = TatsinHttpClient("http://x/tatsin/decide")
    with mock.patch("requests.post", return_value=_Resp(body)):
        return client.decide("state", ["A", "B"], "q")


class Masking(unittest.TestCase):
    def test_email_masked(self):
        out, hits = mask_pii("a@b.com ครับ", {"email": r"[\w.+-]+@[\w-]+\.[\w.]+"},
                             "[REDACTED]")
        self.assertEqual(out, "[REDACTED] ครับ")
        self.assertEqual(hits, ["email:1"])

    def test_thai_digit_run_masked_without_spaces(self):
        # Unspaced Thai: \b would not fire, so the pattern uses digit lookarounds.
        pat = r"(?<!\d)\d[\s-]?\d{4}[\s-]?\d{5}[\s-]?\d{2}[\s-]?\d(?!\d)"
        out, hits = mask_pii("บัตรประชาชน1234567890123ครับ", {"citizen_id": pat},
                             "[REDACTED]")
        self.assertNotIn("1234567890123", out)
        self.assertEqual(hits, ["citizen_id:1"])


class Helpers(unittest.TestCase):
    def test_letter(self):
        self.assertEqual(letter(0), "A")
        self.assertEqual(letter(6), "G")

    def test_parse_reply_reads_json_inside_prose(self):
        self.assertEqual(OpenAIChatClient.parse_reply('ok {"answer":"B","confidence":0.9}'),
                         {"answer": "B", "confidence": 0.9})

    def test_parse_reply_returns_empty_on_garbage(self):
        self.assertEqual(OpenAIChatClient.parse_reply("no json here"), {})

    def test_prompt_lists_every_option(self):
        p = OpenAIChatClient.build_prompt("s", ["ห้อง A", "ห้อง B"], "q")
        self.assertIn("A) ห้อง A", p)
        self.assertIn("B) ห้อง B", p)


class EndpointContract(unittest.TestCase):
    def test_route_with_full_coverage(self):
        d = _decide({"pick": "A", "confidence": 0.9, "noul": 0.1, "margin": 0.8,
                     "coverage": {"scored": 2, "requested": 2}, "source": "stub"})
        self.assertEqual(d.act, "route")
        self.assertEqual(d.pick, "A")
        self.assertEqual(d.noul, 0.1)

    def test_partial_coverage_abstains(self):
        d = _decide({"pick": "A", "coverage": {"scored": 1, "requested": 2}})
        self.assertEqual(d.act, "abstain")
        self.assertEqual(d.error, "partial_coverage:1/2")

    def test_missing_coverage_abstains(self):
        d = _decide({"pick": "A"})
        self.assertEqual(d.act, "abstain")
        self.assertEqual(d.error, "no_coverage")

    def test_missing_pick_abstains(self):
        d = _decide({"coverage": {"scored": 2, "requested": 2}})
        self.assertEqual(d.act, "abstain")
        self.assertEqual(d.error, "no_pick")

    def test_error_field_abstains(self):
        d = _decide({"error": "boom"})
        self.assertEqual(d.act, "abstain")

    def test_transport_error_abstains_not_raises(self):
        client = TatsinHttpClient("http://x")
        with mock.patch("requests.post", side_effect=OSError("down")):
            d = client.decide("s", ["A"], "q")
        self.assertEqual(d.act, "abstain")
        self.assertTrue(d.error.startswith("transport:"))


if __name__ == "__main__":
    unittest.main()
