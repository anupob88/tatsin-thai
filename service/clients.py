"""Model clients for the Tatsin wrapper.

Every client returns a `Decision`. Clients never raise for model-side problems:
a transport error, an unparseable reply, or a server-side error string all come
back as an abstain with `error` set. The ladder then fails *safe* (escalate)
instead of crashing a user request. Root causes still need fixing upstream --
this only stops them from reaching the user.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import requests

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def letter(i: int) -> str:
    return LETTERS[i]


@dataclass
class Decision:
    """One model's answer, normalised across client types."""

    act: str = "route"                 # "route" | "abstain"
    pick: str | None = None            # letter, when the model answered by letter
    confidence: float | None = None    # prob of the pick
    margin: float | None = None        # prob(pick) - prob(runner-up)
    noul: float | None = None          # calibrated abstain head; None when the endpoint omits it
    ranked: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    model: str = ""
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def as_abstain(self, reason: str) -> "Decision":
        return Decision(act="abstain", error=reason, model=self.model,
                        latency_ms=self.latency_ms, raw=self.raw)


def mask_pii(text: str, patterns: dict[str, str], replacement: str) -> tuple[str, list[str]]:
    """Mask PII before anything is logged or stored. Returns (masked, hits)."""
    hits: list[str] = []
    out = text
    for name, pat in patterns.items():
        found = re.findall(pat, out)
        if found:
            hits.append(f"{name}:{len(found)}")
            out = re.sub(pat, replacement, out)
    return out, hits


class StubClient:
    """Scripted client for tests. Pops one scripted Decision per call."""

    def __init__(self, script: Sequence[Decision], model: str = "stub"):
        self._script = list(script)
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def decide(self, state: str, options: Sequence[str], question: str) -> Decision:
        self.calls.append({"state": state, "options": list(options), "question": question})
        if not self._script:
            return Decision(act="abstain", error="stub_exhausted", model=self.model)
        d = self._script.pop(0)
        d.model = self.model
        return d


class JevReaderClient:
    """The `/ask` slot reader -- an OLD internal probe, NOT a Tatsin checkpoint.

    Kept for A/B comparison only. It exposes no `noul`, so the Tatsin 2-layer
    gate cannot run on it, and it treats an out-of-scope message as a hard error
    (POSITION_MISALIGNED) rather than an abstain. Never use it as layer 1 of the
    routing ladder; its numbers say nothing about the trained Tatsin models.
    """

    def __init__(self, url: str, timeout_ms: int = 30000):
        self.url = url
        self.timeout = timeout_ms / 1000
        self.model = "jev-reader"

    def decide(self, state: str, options: Sequence[str], question: str) -> Decision:
        choices = [letter(i) for i in range(len(options))]
        payload = {"state": state, "question": question,
                   "choices": choices, "options": list(options)}
        t0 = time.perf_counter()
        try:
            r = requests.post(self.url, json=payload, timeout=self.timeout)
            body = r.json()
        except Exception as exc:  # transport, empty body, or non-JSON
            return Decision(act="abstain", error=f"transport:{type(exc).__name__}",
                            model=self.model,
                            latency_ms=(time.perf_counter() - t0) * 1000)
        ms = (time.perf_counter() - t0) * 1000

        if not isinstance(body, dict) or body.get("error"):
            # Known trigger: an out-of-scope message makes the reader return
            # POSITION_MISALIGNED instead of abstaining. Treated as abstain so
            # the request escalates rather than erroring at the user.
            return Decision(act="abstain", error=str(body.get("error"))[:200],
                            model=self.model, latency_ms=ms, raw=body)

        ranked = {k: float(v) for k, v in (body.get("ranked_slots") or {}).items()}
        pick = body.get("pick")
        if pick is None or pick not in ranked:
            return Decision(act="abstain", error="no_pick", model=self.model,
                            latency_ms=ms, raw=body)
        return Decision(act="route", pick=pick, confidence=float(body.get("p") or 0.0),
                        margin=float(body.get("margin") or 0.0), ranked=ranked,
                        model=self.model, latency_ms=ms, raw=body)


class TatsinHttpClient:
    """Layer 1: a trained Tatsin checkpoint behind HTTP.

        POST {state, question, options[]}
         -> {pick, confidence, noul, margin, ranked, coverage{scored,requested}}

    `noul` is required for the 2-layer gate; without it we are back to the
    p+margin gate that was measured as *not* ours. `coverage` is required so a
    partially-read option list surfaces instead of silently becoming a score
    (the JeV reader dropped slot G without telling the caller).
    """

    def __init__(self, url: str, timeout_ms: int = 30000, name: str = "tatsin",
                 coverage_policy: str = "abstain"):
        self.url = url
        self.timeout = timeout_ms / 1000
        self.model = name
        self.coverage_policy = coverage_policy

    def decide(self, state: str, options: Sequence[str], question: str) -> Decision:
        payload = {"state": state, "question": question, "options": list(options)}
        t0 = time.perf_counter()
        try:
            r = requests.post(self.url, json=payload, timeout=self.timeout)
            body = r.json()
        except Exception as exc:
            return Decision(act="abstain", error=f"transport:{type(exc).__name__}",
                            model=self.model,
                            latency_ms=(time.perf_counter() - t0) * 1000)
        ms = (time.perf_counter() - t0) * 1000

        if not isinstance(body, dict) or body.get("error"):
            return Decision(act="abstain", error=str(body.get("error"))[:200],
                            model=self.model, latency_ms=ms, raw=body)

        # Which checkpoint answered decides whether `noul` is a real probability
        # (a real abstain head) or a projection of `act` (an act-derived stub). That is the endpoint's fact to
        # report, not ours to assume -- read it off the wire, and only fall back
        # to our own configured name when the body stays silent.
        model = str(body.get("source") or body.get("model") or self.model)

        cov = body.get("coverage")
        scored = cov.get("scored") if isinstance(cov, dict) else None
        requested = cov.get("requested") if isinstance(cov, dict) else None
        complete = (isinstance(scored, int) and isinstance(requested, int)
                    and scored >= requested)
        if not complete and self.coverage_policy == "abstain":
            # `coverage` is mandatory in the endpoint contract, so its absence is
            # a contract violation, not a licence to trust the pick. A short read
            # that stays quiet is exactly how the JeV reader lost slot G.
            reason = (f"partial_coverage:{scored}/{requested}"
                      if isinstance(scored, int) and isinstance(requested, int)
                      else "no_coverage")
            return Decision(act="abstain", model=model, latency_ms=ms,
                            error=reason, raw=body)

        ranked = {k: float(v) for k, v in (body.get("ranked") or {}).items()}
        pick = body.get("pick")
        if pick is None:
            return Decision(act="abstain", model=model, latency_ms=ms,
                            raw=body, error="no_pick")
        noul = body.get("noul")
        return Decision(act="route", pick=pick,
                        confidence=body.get("confidence"),
                        margin=body.get("margin"),
                        noul=float(noul) if isinstance(noul, (int, float)) else None,
                        ranked=ranked, model=model, latency_ms=ms, raw=body)


class OpenAIChatClient:
    """OpenAI-compatible chat endpoint (vLLM / llama.cpp / Ollama).

    Sends the exact prompt shape the small models were trained on:
        สถานะ: <state>
        คำถาม: <question>
        ตัวเลือก: A) <opt> / B) <opt>
        ตอบ JSON สั้นๆ
    and parses {"answer","confidence","act"} out of the reply.
    """

    def __init__(self, base_url: str, model: str, timeout_ms: int = 60000,
                 api_key: str | None = None, max_tokens: int = 64,
                 style: str = "trained"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.timeout = timeout_ms / 1000
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.style = style
        self.model = model

    @staticmethod
    def build_prompt(state: str, options: Sequence[str], question: str) -> str:
        opt_text = " / ".join(f"{letter(i)}) {o}" for i, o in enumerate(options))
        return (f"สถานะ: {state}\nคำถาม: {question}\nตัวเลือก: {opt_text}\nตอบ JSON สั้นๆ")

    @staticmethod
    def build_instruct_prompt(state: str, options: Sequence[str],
                              question: str) -> str:
        """For the big model: explicit task framing, same answer contract."""
        opt_text = "\n".join(f"{letter(i)}) {o}" for i, o in enumerate(options))
        return (
            "คุณเป็นตัวจัดเส้นทางงาน (router) ของบ้าน มีห้องดังต่อไปนี้:\n"
            f"{opt_text}\n\n"
            f"ข้อความที่เข้ามา: {state}\n"
            f"คำถาม: {question}\n\n"
            "เลือกห้องที่ต้องลงมือทำงานนี้จริง ๆ เพียงห้องเดียว "
            "ถ้าข้อความนี้ไม่มีงานในขอบเขตของห้องใดเลย ให้ตอบ abstain\n"
            'ตอบเป็น JSON เท่านั้น: {"answer": "<ตัวอักษร>", "confidence": <0-1>, '
            '"act": "route"|"abstain"}'
        )

    @staticmethod
    def parse_reply(text: str) -> dict[str, Any]:
        m = re.search(r"\{.*?\}", text, re.S)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return {}
        return obj if isinstance(obj, dict) else {}

    def decide(self, state: str, options: Sequence[str], question: str) -> Decision:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        prompt = (self.build_instruct_prompt(state, options, question)
                  if self.style == "instruct"
                  else self.build_prompt(state, options, question))
        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=body,
                              headers=headers, timeout=self.timeout)
            data = r.json()
        except Exception as exc:
            return Decision(act="abstain", error=f"transport:{type(exc).__name__}",
                            model=self.model,
                            latency_ms=(time.perf_counter() - t0) * 1000)
        ms = (time.perf_counter() - t0) * 1000

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            return Decision(act="abstain", error="bad_response_shape",
                            model=self.model, latency_ms=ms, raw=data)

        parsed = self.parse_reply(content)
        if not parsed:
            return Decision(act="abstain", error="unparseable_reply",
                            model=self.model, latency_ms=ms, raw={"content": content})

        act = parsed.get("act", "route")
        answer = parsed.get("answer")
        conf = parsed.get("confidence")
        conf = float(conf) if isinstance(conf, (int, float)) else None
        if act == "abstain" or answer is None:
            return Decision(act="abstain", confidence=conf, model=self.model,
                            latency_ms=ms, raw=parsed)
        if answer not in [letter(i) for i in range(len(options))]:
            return Decision(act="abstain", error=f"answer_out_of_range:{answer}",
                            model=self.model, latency_ms=ms, raw=parsed)
        return Decision(act="route", pick=answer, confidence=conf, model=self.model,
                        latency_ms=ms, raw=parsed)
