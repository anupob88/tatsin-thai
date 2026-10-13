"""Smoke test for the layer-1 server (stdlib only -- no extra deps).

Run the server first (see README.md), then:

    python smoke_test.py
    TATSIN_DECIDE_URL=http://127.0.0.1:8140/tatsin/decide python smoke_test.py
    TATSIN_API_KEY=... python smoke_test.py        # if the server requires a key

Sends the two reference cases from the trainer's QC set:
  * smk_route  -- an in-scope work message -> should pick a room
  * smk_ood    -- a plain greeting (out of scope) -> high noul, i.e. abstain

Reference (raw scorer, no temperature): smk_route pick=A noul~0.056;
smk_ood pick=B noul~0.955. Temperature scaling in rl_agent_config.json shifts
the probabilities slightly -- what must line up is pick + the noul ordering.
"""
import json
import os
import urllib.request

URL = os.environ.get("TATSIN_DECIDE_URL", "http://127.0.0.1:8140/tatsin/decide")
KEY = os.environ.get("TATSIN_API_KEY")

OPTIONS = ["ห้อง code (เบส — dev)", "ห้อง tor (ต่อ — เทรดทอง)",
           "ห้อง brew (ฮอป — ต้มเบียร์)", "ห้อง power (ไฟ — ไฟฟ้ากำลัง)",
           "ห้อง cad (ไหม — 3D design)", "ห้อง data (บูม — วิเคราะห์ข้อมูล)",
           "ห้อง pcb (อาร์ม — วงจร)"]
QUESTION = "ข้อความนี้ควรส่งต่อให้ห้องไหน"

CASES = [
    ("smk_route", "เขียนสคริปต์ Python ดึงข้อมูล API"),
    ("smk_ood", "สวัสดีตอนเย็นครับทุกคน"),
]


def call(state: str) -> dict:
    body = json.dumps({"state": state, "question": QUESTION, "options": OPTIONS},
                      ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["X-Api-Key"] = KEY
    req = urllib.request.Request(URL, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ok = True
    for name, text in CASES:
        try:
            d = call(text)
        except Exception as exc:
            print(f"[{name}] ERROR {type(exc).__name__}: {exc}")
            ok = False
            continue
        cov = d.get("coverage", {})
        full = cov.get("scored") == cov.get("requested")
        print(f"[{name}] pick={d.get('pick')} conf={d.get('confidence')} "
              f"noul={d.get('noul')} margin={d.get('margin')} "
              f"coverage={cov.get('scored')}/{cov.get('requested')} source={d.get('source')}")
        if not full:
            print(f"[{name}] coverage was NOT full -- contract violation")
            ok = False
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
