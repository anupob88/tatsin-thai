"""The Tatsin decision ladder — the wrapper (ชั้นครอบ) around the small models.

Three layers, per the agreed design:
  layer 1  small model decides -> confident => deliver itself
  layer 2  abstain/error       -> escalate to the big model -> it delivers
  layer 3  big model unsure    -> quarantine room (no question back to the user)

Nothing here needs a human confirmation step: the user types, the system routes.

N-way note: the small models were trained on *pairwise* A/B prompts only
(build_qwen_v3.py). `nway.strategy = "pairwise"` keeps every model call inside
that distribution and cancels the measured position bias by running each pair
twice with the options swapped. `strategy = "single"` is one call over all rooms
(cheap, off-distribution) and exists so the two can be measured against each
other before either is shipped.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from clients import Decision, mask_pii, letter

DEFAULT_QUESTION = "ข้อความนี้ควรส่งต่อให้ห้องไหน"
DEFAULT_QUARANTINE_ID = "quarantine"
DEFAULT_QUARANTINE_LABEL = "ห้องกัก"

# "no answer supplied" is not the same as "deliberately no answer": a quarantine
# result has a destination but nothing to say to the user.
_UNSET: Any = object()

# Identity matching: the message must BE the identity question, not merely
# contain or end with it. These are peeled off both ends first.
#
# "หน่อย" is deliberately absent: 'แนะนำตัวหน่อย' uses it as its own pin, so
# peeling it would turn a work message into the keyword.
_TRAILING_PARTICLES = ("ครับ", "คร้าบ", "ค่ะ", "คะ", "คับ", "จ้ะ", "จ๊ะ", "นะ",
                       "อ่ะ", "อะ", "จ้า", "จ๋า", "น้า", "สิ", "ซิ", "เนอะ",
                       "แหละ", "บ้าง", "though", "then", "anyway", "please")
_TRAILING_PUNCT = " \t\r\n.!?,;:…~^*\"'“”‘’`_-—–"

# A plain identity question may open politely. Anything else in front of the
# keyword means the phrase is quoted or embedded content, not the question:
# 'แก้โค้ดตรงที่ถามว่าคุณชื่ออะไร' is a code change that quotes the question, and
# answering it with the brand sentence is a wrong answer with zero model calls.
_LEADINS = ("ช่วยบอก", "ช่วย", "รบกวน", "ขอสอบถาม", "ขอถาม", "อยากถาม", "อยากรู้",
            "อยากทราบ", "ขอ", "ถาม", "please", "hey", "hi", "hello", "so", "ok",
            "okay", "just")

# Longest first, or a shorter lead-in that is a prefix of a longer one swallows
# it: 'ok' would take 'okay คุณคือใคร' down to 'ay คุณคือใคร', which peels no
# further and loses an identity question that should not have gone to the model.
_LEADINS_BY_LEN = tuple(sorted(_LEADINS, key=len, reverse=True))


def _strip_tail(text: str) -> str:
    """Peel trailing punctuation and politeness particles, repeatedly."""
    out = text.strip()
    while True:
        before = out
        out = out.rstrip(_TRAILING_PUNCT).strip()
        low = out.lower()
        for particle in _TRAILING_PARTICLES:
            if low.endswith(particle):
                out = out[: -len(particle)]
                break
        if out == before:
            return out


def _identity_core(text: str) -> str:
    """Reduce a message to what is left once politeness is peeled off both ends.

    Returns the whole (stripped) message when nothing peels, which is what makes
    the caller's exact compare reject quoted or embedded phrases.
    """
    head = _strip_tail(text)
    while head:
        for lead in _LEADINS_BY_LEN:
            if head.lower().startswith(lead):
                rest = _strip_tail(head[len(lead):])
                if rest and rest != head:
                    head = rest
                    break
        else:
            return head
    return ""


@dataclass(frozen=True)
class Room:
    id: str
    label: str
    channel_id: str = ""


@dataclass
class Config:
    rooms: list[Room]
    identity: dict[str, Any]
    pdpa: dict[str, Any]
    gates: dict[str, Any]
    nway: dict[str, Any]
    endpoints: dict[str, Any]
    timeouts: dict[str, Any]
    coverage_policy: str = "abstain"
    quarantine: Room = field(
        default_factory=lambda: Room(DEFAULT_QUARANTINE_ID, DEFAULT_QUARANTINE_LABEL))

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        q = raw.get("quarantine") or {}
        return Config(
            rooms=[Room(r["id"], r["label"], r.get("channel_id", ""))
                   for r in raw["rooms"]],
            identity=raw.get("identity", {}),
            pdpa=raw.get("pdpa", {}),
            gates=raw.get("gates", {}),
            nway=raw.get("nway", {}),
            endpoints=raw.get("endpoints", {}),
            timeouts=raw.get("timeouts", {}),
            coverage_policy=raw.get("coverage_policy", "abstain"),
            quarantine=Room(q.get("id", DEFAULT_QUARANTINE_ID),
                            q.get("label", DEFAULT_QUARANTINE_LABEL),
                            q.get("channel_id", "")),
        )

    def room_by_letter(self, lt: str) -> Room | None:
        i = letter_index(lt)
        if i is None or i >= len(self.rooms):
            return None
        return self.rooms[i]


def letter_index(lt: str) -> int | None:
    if not lt or len(lt) != 1:
        return None
    i = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(lt.upper())
    return i if i >= 0 else None


@dataclass
class Step:
    """One auditable step of the ladder."""

    stage: str
    detail: dict[str, Any] = field(default_factory=dict)


class Ladder:
    def __init__(self, cfg: Config, small, big=None):
        self.cfg = cfg
        self.small = small
        self.big = big
        self._calls = 0

    # ---------- layer 1: small model ----------

    def _pairwise(self, state: str, question: str, a: Room, b: Room,
                  budget: int) -> tuple[Room | None, list[Decision], str | None]:
        """Ask a vs b, then again with the options swapped.

        Returns (winner, decisions, note). `note` is None when the pair was
        decisive, otherwise the reason no winner could be taken.

        A pair is only trustworthy WITH its swap (that is what cancels the
        measured position bias), so a pair costs 2 slots and is skipped whole
        when fewer than 2 remain -- never half-run, never over budget.
        """
        if self._calls + 2 > budget:
            return None, [], "budget"
        d1 = self.small.decide(state, [a.label, b.label], question)
        self._calls += 1
        if d1.act != "route" or d1.pick is None:
            return None, [d1], "abstain"
        win1 = a if d1.pick == "A" else b
        if not self.cfg.nway.get("swap_check", True):
            return win1, [d1], None

        d2 = self.small.decide(state, [b.label, a.label], question)
        self._calls += 1
        if d2.act != "route" or d2.pick is None:
            return None, [d1, d2], "abstain"
        win2 = b if d2.pick == "A" else a
        if win1 is not win2:
            # Same pair, opposite answers -> position bias, not a preference.
            return None, [d1, d2], "swap_disagreement"
        return win1, [d1, d2], None

    def _choose_single(self, state: str, question: str,
                       budget: int) -> tuple[Room | None, list[Decision]]:
        if self._calls >= budget:
            return None, []
        d = self.small.decide(state, [r.label for r in self.cfg.rooms], question)
        self._calls += 1
        if d.act != "route" or d.pick is None:
            return None, [d]
        return self.cfg.room_by_letter(d.pick), [d]

    def _choose_pairwise(self, state: str, question: str, budget: int
                         ) -> tuple[Room | None, list[Decision], int, Decision | None]:
        """Single-elimination tournament, seeded by one N-way call.

        The seed is off-distribution so it only *proposes*; every deciding call
        is a pairwise A/B, which is what the model was trained on.
        """
        seed, seed_decisions = self._choose_single(state, question, budget)
        seed_decision = seed_decisions[0] if seed_decisions else None
        if seed_decision is not None and seed_decision.error:
            # `error` means the endpoint could not answer this request at all --
            # unreachable, or violating the contract (no coverage). A model that
            # *chose* to abstain sets no error. Either way, one call is enough:
            # escalating beats hammering a broken endpoint N more times.
            return None, [], self.cfg.gates.get("pairwise_abstain_votes", 3), seed_decision
        if seed_decision is None:
            # No call was made at all: the budget was too small to even seed the
            # tournament. There is no evidence to route on, so do NOT fall through
            # to the rooms[0] default below -- that would route with zero model
            # calls, a silent misroute in a router meant to abstain, not guess.
            return None, [], 0, None
        champion = seed or self.cfg.rooms[0]
        wins: list[Decision] = []
        abstain_votes = 0 if seed is not None else 1
        for challenger in self.cfg.rooms:
            if challenger is champion:
                continue
            if self._calls + 2 > budget:
                break
            winner, ds, note = self._pairwise(state, question, champion, challenger, budget)
            if note is not None:
                abstain_votes += 1      # abstain OR position-inconsistent: no signal
                continue
            champion = winner
            wins.extend(ds)
        return champion, wins, abstain_votes, seed_decision

    def _gate_small(self, wins: Sequence[Decision], abstain_votes: int,
                    calls: int) -> tuple[bool, str, dict[str, Any]]:
        g = self.cfg.gates
        if calls == 0:
            return True, "no_calls", {}
        if abstain_votes >= g.get("pairwise_abstain_votes", 3):
            return False, f"abstain_votes={abstain_votes}", {"abstain_votes": abstain_votes}
        if not wins:
            return False, "no_wins", {"abstain_votes": abstain_votes}

        # Only some checkpoints emit a real `noul` probability. Others (Qwen)
        # project `act` into it (abstain=1.0 / answer=0.0), and gating on that
        # is just reading `act` through a thicker name.
        exact = set(g.get("noul_is_exact_by_model", []))
        listed = [d for d in wins if d.model in exact]
        nouls = [d.noul for d in listed if d.noul is not None]
        if nouls:
            # 2-layer noul gate. Both layers must be present: layer 1 alone would
            # miss t0016, layer 2 alone would over-abstain 4 correct routes.
            noul = max(nouls)
            margins = [d.margin for d in wins if d.margin is not None]
            margin = min(margins) if margins else 0.0
            info = {"noul": noul, "margin": margin}
            if noul >= g.get("noul_layer1", 0.5):
                return False, "layer1_noul", info
            if noul > g.get("noul_layer2", 0.3) and margin < g.get("noul_layer2_margin", 0.5):
                return False, "layer2_noul_margin", info
            return True, "gate_pass", info

        if listed:
            # A declared noul-exact model answered without a `noul`. Falling
            # through to the confidence gate here would swap the ruler for one we
            # measured as *not* ours -- and do it silently, producing
            # plausible-looking numbers instead of an error. Abstain instead.
            who = ",".join(sorted({d.model for d in listed}))
            return False, f"noul_missing:{who}", {"noul_missing": who}

        confs = [d.confidence for d in wins if d.confidence is not None]
        if not confs:
            return False, "no_confidence", {}
        lo = min(confs)
        if lo < g.get("small_min_confidence", 0.5):
            return False, "low_confidence", {"min_confidence": lo}
        return True, "confidence_pass", {"min_confidence": lo}

    def _small_budget(self) -> int:
        """Small-layer call cap for one request.

        `nway.max_calls` is the TOTAL budget for the request, not the small
        layer's alone: when a big model is configured, one slot is reserved for
        its escalation so the request can never spend more than `max_calls`
        calls end to end. Without a big model the small layer may use them all.
        """
        total = int(self.cfg.nway.get("max_calls", 16))
        return max(0, total - (1 if self.big is not None else 0))

    def _layer_small(self, state: str, question: str) -> dict[str, Any]:
        # `calls` means "what this decision cost", not "what this process has
        # spent". `decide()` resets the counter itself, but callers that come in
        # one level down (`_layer_small` directly -- the game harness, the gate
        # board) never did, so their counter read as a running process total and
        # grew across turns. Reset here so the number is the same regardless of
        # which entry point asked.
        self._calls = 0
        budget = self._small_budget()
        strategy = self.cfg.nway.get("strategy", "pairwise")
        if strategy == "single":
            room, decisions = self._choose_single(state, question, budget)
            wins = [d for d in decisions if d.act == "route"]
            abstain_votes = sum(1 for d in decisions if d.act != "route")
        else:
            room, wins, abstain_votes, _seed = self._choose_pairwise(state, question, budget)
        ok, reason, info = self._gate_small(wins, abstain_votes, self._calls)
        return {"room": room, "ok": ok, "reason": reason, "info": info,
                "strategy": strategy, "calls": self._calls}

    # ---------- public ----------

    def decide(self, state: str, question: str = DEFAULT_QUESTION,
               source: str = "web") -> dict[str, Any]:
        t0 = time.perf_counter()
        self._calls = 0
        trace: list[Step] = []

        masked, pii_hits = mask_pii(state, self.cfg.pdpa.get("patterns", {}),
                                    self.cfg.pdpa.get("mask", "[REDACTED]"))
        trace.append(Step("mask", {"pii_hits": pii_hits}))

        ident = self._identity(masked)
        if ident is not None:
            trace.append(Step("identity", {}))
            return self._result(None, None, "identity", "route",
                                {"reason": "identity_short_circuit"}, trace, source,
                                t0, answer=ident)

        small = self._layer_small(masked, question)
        trace.append(Step("layer1_small", {k: v for k, v in small.items()
                                           if k != "room"}))
        if small["ok"] and small["room"] is not None:
            return self._result(small["room"], small["info"], "small", "route",
                                {"reason": small["reason"], "strategy": small["strategy"],
                                 "calls": small["calls"]}, trace, source, t0)

        if self.big is not None and self._calls < int(self.cfg.nway.get("max_calls", 16)):
            d = self.big.decide(masked, [r.label for r in self.cfg.rooms], question)
            self._calls += 1
            room = self.cfg.room_by_letter(d.pick) if d.pick else None
            conf = d.confidence or 0.0
            trace.append(Step("layer2_big", {"act": d.act, "pick": d.pick,
                                             "confidence": conf, "error": d.error}))
            if d.act == "route" and room is not None and conf >= self.cfg.gates.get("big_min_confidence", 0.6):
                return self._result(room, {"confidence": conf, "noul": d.noul},
                                    "big", "route",
                                    {"reason": "big_confident", "escalated_from": small["reason"]},
                                    trace, source, t0)

        trace.append(Step("layer3_quarantine", {}))
        return self._result(self.cfg.quarantine, None, "quarantine", "abstain",
                            {"reason": "no_layer_confident",
                             "layer1_reason": small["reason"]},
                            trace, source, t0, answer=None)

    def _identity(self, text: str) -> str | None:
        """Answer the identity question ourselves, or return None to route.

        The message must BE the identity question -- the keyword alone, or behind
        a polite lead-in like 'ช่วยแนะนำตัวหน่อย' -- so cheating needs no more
        than an exact-string compare. Two weaker rules were measured and both
        hijacked real work, each answering with the brand sentence and zero model
        calls instead of routing:

          * substring containment: 'who are u' is a prefix of 'who are u**sing**',
            so 'the interns who are using the GPU need access' answered itself.
          * trailing anchor: quoting the question ends with it, so
            'แก้โค้ดตรงที่ถามว่าคุณชื่ออะไร' and 'fix the test that asserts who are u'
            answered themselves.

        Accepted trade-off: a message that asks the identity question *and*
        something else ('who are you? I need help') routes -- the request for
        work is the actionable part, and routing it is not a wrong answer.
        """
        ident = self.cfg.identity
        core = _identity_core(text)
        if not core:
            return None
        low = core.lower()
        for kw in ident.get("keywords_th", []):
            if core == kw:
                return ident.get("answer_th")
        for kw in ident.get("keywords_en", []):
            if low == kw.lower():
                return ident.get("answer_en")
        return None

    def _result(self, room: Room | None, info: dict | None, layer: str, act: str,
                gate: dict, trace: list[Step], source: str,
                t0: float, answer: Any = _UNSET) -> dict[str, Any]:
        return {
            "route": room.id if room else None,
            "room_label": room.label if room else None,
            "channel_id": room.channel_id if room else None,
            "answer": (room.label if room else None) if answer is _UNSET else answer,
            "act": act,
            "layer": layer,
            "gate": gate,
            "signal": info or {},
            "calls": self._calls,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "source": source,
            "trace": [asdict(s) for s in trace],
        }
