# ตัวเสิร์ฟโมเดลเล็ก (ชั้น 1) — `/tatsin/decide`

ตัวนี้คือ "ปลั๊ก" ระหว่าง checkpoint ของตัดสิน กับชั้นครอบ (`../service/`)
มันโหลด checkpoint แล้วเปิด HTTP ตามสัญญาที่ชั้นครอบคาดไว้ (ดู `../README.md` ข้อ 6)

> **แยกส่วนจากชั้นครอบ:** โฟลเดอร์ `model_server/` = โค้ดโหลด+เสิร์ฟโมเดล
> `../service/` = ตรรกะตัดสิน (PDPA · ตัวตน · เกต 3 ชั้น · API ผู้ใช้)

---

## 1. น้ำหนักโมเดล (checkpoint)

น้ำหนัก **ไม่ได้อยู่ในรีโปนี้** (ไฟล์ใหญ่ ~1.3 GB) — ดาวน์โหลดแยกแล้ววางเป็น:

```
model_server/ckpt/tatsin_v2/
├── model.safetensors          ← น้ำหนักจริง (1.29 GB)
├── rl_agent_config.json       ← บันทึก base model + สถาปัตยกรรมหัว
├── encoder/config.json        ← โครงเอนจิน (mmBERT-base)
├── tokenizer/tokenizer.json
├── tokenizer/tokenizer_config.json
└── README.md                  ← การ์ดต้นทาง (เครดิตเอนจินตั้งต้น)
```

> **ชื่อโฟลเดอร์ = `source` ที่โมเดลรายงาน** (`tatsin_v2` โดยปริยาย) ชั้นครอบใช้ค่านี้
> เทียบกับ `gates.noul_is_exact_by_model` — **ถ้าชื่อไม่ตรง เกต `noul` 2 ชั้นจะเงียบ ๆ
> เลื่อนไปใช้เกต `confidence` แทน** ถ้าใช้ชื่อโฟลเดอร์อื่น ให้ตั้ง `TATSIN_SOURCE_TAG`
> ให้ตรง หรือแก้ `noul_is_exact_by_model` ใน `../service/config.json`

**ที่มา/สัญญา:** checkpoint นี้เป็น **ผลงานดัดแปลง** ต่อยอดจากเอนจินตัดสินใจโอเพ่นซอร์ส
**Laya Multilingual (mmBERT-base, 322M)** โดย **Convai Innovations** ภายใต้ **Apache-2.0**
— ดูเครดิตเต็มที่ [`../NOTICE`](../NOTICE) และ [`../MODEL_CARD.md`](../MODEL_CARD.md) §8
(ไฟล์ `README.md` ต้นทางในตัว checkpoint คงไว้ตามข้อกำหนดสัญญา §4c)

---

## 2. ติดตั้ง

```bash
cd model_server
pip install -r requirements.txt      # torch + transformers + safetensors + numpy
```

> **ไม่ต้อง `pip install laya`** — โค้ดเสิร์ฟ (`rl_agent_api.py`) สร้างเอนจินด้วย
> `transformers` ตรง ๆ แล้วโหลด state dict ที่บันทึกไว้ แพ็กเกจ `laya` เป็นแค่
> API สะดวกของเอนจินตั้งต้น ไม่ได้ถูกเรียกใช้ที่นี่
>
> `decide_endpoint.py` ตั้ง `USE_TF=0` เอง — `transformers` ลอง import TensorFlow
> ตอนโหลด และ abseil runtime ของ TF ทำโมเดลค้างได้

**ทรัพยากร:** RAM/commit อย่างน้อย **~4 GB** ว่าง (โมเดล ~1.3 GB + ตอนประกอบอีก ~2 GB)
· ค่าเริ่มต้นรันบน **CPU** (ช้ากว่า GPU แต่พอใช้ — median ~47 ms/คำถามบนเครื่องผู้พัฒนา)

---

## 3. รัน

```bash
python decide_endpoint.py                     # 127.0.0.1:8140 (ค่าเริ่มต้น ปลอดภัย)
python decide_endpoint.py --port 9000         # เปลี่ยนพอร์ต
```

เช็คว่าขึ้นแล้ว:

```bash
curl -s localhost:8140/health
# {"ok":true,"source":"tatsin_v2","ckpt":"tatsin_v2","device":"cpu","load_s":..,"commit_headroom_gb_now":..}
```

จากนั้นชี้ชั้นครอบไปที่มัน (`../service/`):

```bash
export TATSIN_SMALL_URL="http://127.0.0.1:8140/tatsin/decide"
export TATSIN_SMALL_NAME="tatsin_v2"     # ตรงกับ noul_is_exact_by_model
```

---

## 4. ทดสอบ (smoke)

เปิดเซิร์ฟเวอร์ค้างไว้ แล้วรัน (stdlib ล้วน ไม่ต้องลงอะไรเพิ่ม):

```bash
python smoke_test.py
# [smk_route] pick=A conf=.. noul=~0.056 margin=.. coverage=7/7 source=tatsin_v2
# [smk_ood]   pick=B conf=.. noul=~0.955 margin=.. coverage=7/7 source=tatsin_v2
# OK
```

- `smk_route` — ข้อความงานในขอบเขต → เลือกห้อง
- `smk_ood` — คำทักทาย (นอกขอบเขต) → `noul` สูง = **งด** (นี่คือพฤติกรรมที่ถูก)
- `coverage=7/7` — โมเดลอ่านตัวเลือกครบทุกตัว (ไม่ครบ = ผิดสัญญา)

---

## 5. สัญญา HTTP

**คำขอ:** `POST /tatsin/decide`
```json
{"state": "ข้อความผู้ใช้", "question": "ข้อความนี้ควรส่งต่อให้ห้องไหน", "options": ["ห้อง A", "ห้อง B", "..."]}
```
**ตอบ:** `{pick, confidence, noul, margin, ranked, coverage{scored,requested}, source, latency_ms}`

- **`noul`** และ **`coverage`** บังคับ (เหตุผลอยู่ใน `../README.md` §6)
- สูงสุด **8 ตัวเลือก (A–H)**; ถ้าตัวเลือกยาวจนไม่พอดีหัว → ตอบ **422** พร้อม `coverage{scored:0}`
  (ไม่ตัดตัวเลือกทิ้งเงียบ ๆ)
- `state` ถูกห่อด้วยเฟรมตอนเทรน `พ่อส่งข้อความเข้ามา: "…"` — **ผูกกับโมเดล อย่าแก้ถ้อยคำ**

---

## 6. env ทั้งหมด

| env | หน้าที่ | ค่าเริ่มต้น |
|-----|---------|-------------|
| `TATSIN_CKPT` | โฟลเดอร์ checkpoint | `model_server/ckpt/tatsin_v2` |
| `TATSIN_SOURCE_TAG` | ค่า `source` ที่รายงาน | ชื่อโฟลเดอร์ checkpoint |
| `TATSIN_HOST` / `TATSIN_PORT` | ที่อยู่ที่ฟัง | `127.0.0.1` / `8140` |
| `TATSIN_API_KEY` | กุญแจรับเข้า (บังคับเมื่อเปิดออกเครือข่าย) | ไม่ตั้ง = ไม่ต้องใช้ (เฉพาะ loopback) |
| `TATSIN_DEVICE` | `cpu` หรือ `cuda` | `cpu` |
| `TATSIN_MIN_COMMIT_GB` | ยกเลิกการรันถ้า commit ว่างต่ำกว่านี้ | `2.5` |
| `TATSIN_SAFETENSORS_BACKEND` | `pread` หรือ `mmap` | `pread` |

---

## 7. ความปลอดภัย

- **ค่าเริ่มต้นฟัง `127.0.0.1` เท่านั้น** — เครื่องอื่นในวงเข้าไม่ได้
- ถ้าจะเปิดออกเครือข่าย (`--host 0.0.0.0`) → ตัวรัน **ปฏิเสธ** จนกว่าจะตั้ง
  `TATSIN_API_KEY` (เพราะ endpoint ที่ไม่มีการยืนยันตัวตน เปิดให้ใครก็ได้ยิงข้อความ
  เข้าระบบ) — ตั้งแล้วคำขอ `POST` ต้องแนบ header `X-Api-Key`
- จำกัดขนาด body (1 MB) และจำนวนตัวเลือก (8) ก่อนแตะโมเดล

---

## 8. หน่วยความจำตึง / Windows

`decide_endpoint.py` อ่าน state dict ด้วย backend **`pread`** โดยปริยาย — บน Windows
backend `mmap` (ค่าปริยายของ safetensors) จะ map ไฟล์ทั้งก้อนแล้ว **crash ด้วย access
violation** เมื่อ commit เหลือน้อย (~1.8 GB) ส่วน `pread` อ่าน bytes ตรง ๆ และโหลด
ไฟล์เดิมได้ด้วย headroom ที่ต่ำกว่ามาก ถ้าเครื่องโล่งและอยากได้ mmap กลับ ใช้
`TATSIN_SAFETENSORS_BACKEND=mmap`

ก่อนโหลด ตัวรันจะเช็ค commit headroom และ **ยกเลิก** ถ้าต่ำกว่า `TATSIN_MIN_COMMIT_GB`
(ปริยาย 2.5 GB) — กันโหลดค้างครึ่งทางบนเครื่องที่พื้นที่ไม่พอ
