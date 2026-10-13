#!/usr/bin/env bash
# ตัวอย่างเรียกใช้ Tatsin router ด้วย curl
#
# รัน:  bash examples/decide_curl.sh
# ปรับ BASE / KEY ให้ตรงกับที่รัน service (ค่าเริ่มต้น localhost:8130)

set -euo pipefail

BASE="${TATSIN_HOST:-http://localhost:8130}"
KEY="${TATSIN_API_KEY:-}"

# หมายเหตุ: ใช้ --data-binary @ไฟล์ (หรือ heredoc) — อย่าส่ง -d inline
# เพราะ shell จะกินอักขระ UTF-8 ไทยแล้วข้อความเพี้ยน

echo "== 1) สุขภาพ service =="
curl -s "$BASE/health"; echo

echo
echo "== 2) ตารางห้อง =="
curl -s "$BASE/rooms"; echo

echo
echo "== 3) ตัดสินข้อความ (จากไฟล์ตัวอย่าง) =="
if [ -n "$KEY" ]; then
  curl -s -X POST "$BASE/decide" \
    -H 'Content-Type: application/json' \
    -H "X-Api-Key: $KEY" \
    --data-binary @examples/sample_esp32.json
else
  curl -s -X POST "$BASE/decide" \
    -H 'Content-Type: application/json' \
    --data-binary @examples/sample_esp32.json
fi
echo

echo
echo "== 4) ตัดสินจากข้อความสั่งตรง (heredoc, UTF-8 ปลอดภัย) =="
curl -s -X POST "$BASE/decide" \
  -H 'Content-Type: application/json' \
  ${KEY:+-H "X-Api-Key: $KEY"} \
  --data-binary @- <<'JSON'
{"state":"อยากได้สูตรต้ม IPA ให้ขมกำลังดี","question":"ข้อความนี้ควรส่งต่อให้ห้องไหน","source":"curl-demo"}
JSON
echo
