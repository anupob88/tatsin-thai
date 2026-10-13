"""Behavioural tests for the routing ladder.

Runs with the Python standard library only (no weights, no network):

    cd service
    python -m unittest discover -s tests -v

The clients are `StubClient` instances, so every model call is scripted and the
budget accounting is exact.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients import Decision, StubClient  # noqa: E402
from ladder import Config, Ladder, Room  # noqa: E402

ROOMS = [Room("code", "ห้อง code"), Room("tor", "ห้อง tor"), Room("brew", "ห้อง brew"),
         Room("power", "ห้อง power"), Room("cad", "ห้อง cad"), Room("data", "ห้อง data"),
         Room("pcb", "ห้อง pcb")]

IDENTITY = {"keywords_th": ["คุณคือใคร"], "answer_th": "ผมคือ ตัดสิน (Tatsin)",
            "keywords_en": ["who are you"], "answer_en": "I am Tatsin."}
PDPA = {"patterns": {"email": r"[\w.+-]+@[\w-]+\.[\w.]+"}, "mask": "[REDACTED]"}


def make_config(strategy: str = "single", max_calls: int = 16,
                gates: dict | None = None) -> Config:
    g = {"noul_is_exact_by_model": ["stub"], "small_min_confidence": 0.5,
         "noul_layer1": 0.5, "noul_layer2": 0.3, "noul_layer2_margin": 0.5,
         "pairwise_abstain_votes": 3, "big_min_confidence": 0.6}
    if gates:
        g.update(gates)
    return Config(rooms=ROOMS, identity=IDENTITY, pdpa=PDPA, gates=g,
                  nway={"strategy": strategy, "swap_check": True, "max_calls": max_calls},
                  endpoints={}, timeouts={})


def route(pick: str, *, noul: float | None = 0.1, margin: float = 0.8,
          confidence: float = 0.9, model: str = "stub") -> Decision:
    return Decision(act="route", pick=pick, confidence=confidence, margin=margin,
                    noul=noul, model=model)


class IdentityAndPii(unittest.TestCase):
    def test_identity_answers_without_any_model_call(self):
        small = StubClient([])
        out = Ladder(make_config(), small).decide("คุณคือใคร")
        self.assertEqual(out["layer"], "identity")
        self.assertEqual(out["calls"], 0)
        self.assertEqual(out["answer"], IDENTITY["answer_th"])
        self.assertEqual(small.calls, [])

    def test_embedded_identity_question_is_routed_not_answered(self):
        # Quoting the question is not asking it.
        small = StubClient([route("A")])
        out = Ladder(make_config(), small).decide("แก้โค้ดตรงที่ถามว่าคุณชื่ออะไร")
        self.assertNotEqual(out["layer"], "identity")
        self.assertEqual(len(small.calls), 1)

    def test_pii_is_masked_before_the_model_sees_it(self):
        small = StubClient([route("A")])
        Ladder(make_config(), small).decide("ติดต่อ somchai@example.com ครับ")
        seen = small.calls[0]["state"]
        self.assertNotIn("somchai@example.com", seen)
        self.assertIn("[REDACTED]", seen)


class NoulGate(unittest.TestCase):
    def test_high_noul_abstains_layer1(self):
        small = StubClient([route("A", noul=0.6)])
        out = Ladder(make_config(), small).decide("งานอะไรสักอย่าง")
        self.assertEqual(out["layer"], "quarantine")
        self.assertEqual(out["gate"]["layer1_reason"], "layer1_noul")

    def test_mid_noul_narrow_margin_abstains_layer2(self):
        small = StubClient([route("A", noul=0.4, margin=0.2)])
        out = Ladder(make_config(), small).decide("งานอะไรสักอย่าง")
        self.assertEqual(out["layer"], "quarantine")
        self.assertEqual(out["gate"]["layer1_reason"], "layer2_noul_margin")

    def test_mid_noul_wide_margin_passes(self):
        small = StubClient([route("A", noul=0.4, margin=0.9)])
        out = Ladder(make_config(), small).decide("งานอะไรสักอย่าง")
        self.assertEqual(out["layer"], "small")
        self.assertEqual(out["gate"]["reason"], "gate_pass")

    def test_declared_noul_model_without_noul_abstains(self):
        small = StubClient([route("A", noul=None, model="stub")])
        out = Ladder(make_config(), small).decide("งานอะไรสักอย่าง")
        self.assertEqual(out["layer"], "quarantine")
        self.assertEqual(out["gate"]["layer1_reason"], "noul_missing:stub")

    def test_unlisted_model_falls_back_to_confidence_gate(self):
        # StubClient stamps its own name onto every Decision, so the model name
        # is set on the client, not on the scripted Decision.
        small = StubClient([route("A", noul=None, confidence=0.9)], model="other")
        out = Ladder(make_config(), small).decide("งานอะไรสักอย่าง")
        self.assertEqual(out["gate"]["reason"], "confidence_pass")

        small2 = StubClient([route("A", noul=None, confidence=0.4)], model="other")
        out2 = Ladder(make_config(), small2).decide("งานอะไรสักอย่าง")
        self.assertEqual(out2["layer"], "quarantine")
        self.assertEqual(out2["gate"]["layer1_reason"], "low_confidence")


class LayerTwo(unittest.TestCase):
    def _abstaining_small(self):
        # Always answering "A" makes every swapped pair disagree -> no winner.
        return StubClient([route("A")] * 40)

    def test_big_confident_delivers(self):
        big = StubClient([route("B", confidence=0.9)])
        out = Ladder(make_config("pairwise"), self._abstaining_small(), big) \
            .decide("งานอะไรสักอย่าง")
        self.assertEqual(out["layer"], "big")
        self.assertEqual(out["route"], "tor")

    def test_big_unsure_quarantines(self):
        big = StubClient([route("B", confidence=0.3)])
        out = Ladder(make_config("pairwise"), self._abstaining_small(), big) \
            .decide("งานอะไรสักอย่าง")
        self.assertEqual(out["layer"], "quarantine")
        self.assertEqual(out["act"], "abstain")

    def test_no_big_quarantines_directly(self):
        out = Ladder(make_config(), StubClient([route("A", noul=0.9)])).decide("x")
        self.assertEqual(out["layer"], "quarantine")


class Budget(unittest.TestCase):
    """nway.max_calls is the TOTAL per-request cap, the big layer included."""

    def _run(self, max_calls: int, with_big: bool) -> int:
        small = StubClient([route("A")] * 40)   # forces swap disagreement
        big = StubClient([route("B", confidence=0.3)] * 4) if with_big else None
        out = Ladder(make_config("pairwise", max_calls=max_calls), small, big) \
            .decide("งานอะไรสักอย่าง")
        self.assertLessEqual(out["calls"], max_calls)
        return out["calls"]

    def test_never_exceeds_with_big(self):
        for cap in (1, 2, 3, 4, 5, 8, 16):
            self._run(cap, with_big=True)

    def test_never_exceeds_without_big(self):
        for cap in (1, 2, 3, 5, 16):
            self._run(cap, with_big=False)

    def test_big_still_runs_under_a_tight_cap(self):
        # cap 4 -> small budget 3 (one slot reserved), so the escalation fits.
        self.assertEqual(self._run(4, with_big=True), 4)

    def test_too_small_a_budget_abstains_instead_of_routing_to_room_a(self):
        # A cap that cannot even afford the seed call must NOT fall through to the
        # `rooms[0]` default -- that routed to the first room with zero model
        # calls. With no call there is no evidence, so it must abstain.
        small = StubClient([route("A")] * 8)
        out = Ladder(make_config("pairwise", max_calls=0), small).decide("งานอะไรสักอย่าง")
        self.assertEqual(small.calls, [])
        self.assertEqual(out["act"], "abstain")
        self.assertEqual(out["layer"], "quarantine")
        self.assertEqual(out["route"], "quarantine")   # NOT the first room


if __name__ == "__main__":
    unittest.main()
