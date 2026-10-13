#!/usr/bin/env python3
"""ตัวอย่างเรียกใช้ Tatsin router ด้วย Python (ไม่ต้องใช้ SDK ใด ๆ)

รัน:  python examples/decide_python.py
ปรับ BASE / KEY ให้ตรงกับที่รัน service (ค่าเริ่มต้น localhost:8130)
"""
from __future__ import annotations

import json
import os
import urllib.request

BASE = os.environ.get("TATSIN_HOST", "http://localhost:8130")
KEY = os.environ.get("TATSIN_API_KEY", "")


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **({"X-Api-Key": KEY} if KEY else {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def explain(out: dict) -> str:
    """อ่านผลลัพธ์แบบคน: ส่งไปไหน ชั้นไหน ทำไมถึงงด"""
    if out.get("act") == "abstain":
        return (f"งด → ห้องกัก ({out.get('room_label')}) "
                f"| ชั้น {out.get('layer')} | เหตุผล: {out.get('gate', {}).get('reason')}")
    return (f"ส่งไป «{out.get('room_label')}» (id={out.get('route')}) "
            f"| ชั้น {out.get('layer')} | calls={out.get('calls')} "
            f"| {out.get('latency_ms')} ms | gate={out.get('gate', {}).get('reason')}")


MESSAGES = [
    "เขียนเฟิร์มแวร์ ESP32 อ่านค่าเซ็นเซอร์อุณหภูมิ NTC แล้วส่งขึ้น WiFi",   # คาดว่า → code
    "อยากได้สูตรต้ม IPA ให้ขมกำลังดี",                                     # คาดว่า → brew
    "ช่วยวิเคราะห์ยอดขายเดือนที่แล้วเป็นกราฟให้หน่อย",                       # คาดว่า → data
    "วันนี้อยากได้กาแฟ",                                                   # คาดว่า → งด/กัก (นอกขอบเขต)
    "คุณคือใคร",                                                           # คาดว่า → ประโยคตัวตน (0 call)
]


def main() -> None:
    print("health :", json.dumps(get("/health"), ensure_ascii=False))
    print()
    for msg in MESSAGES:
        out = post("/decide", {"state": msg, "source": "python-demo"})
        print(f"• {msg}")
        print(f"    → {explain(out)}")
        if out.get("answer"):
            print(f"    answer: {out['answer']}")
        print()


if __name__ == "__main__":
    main()
