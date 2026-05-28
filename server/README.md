# Faster Whisper FastAPI Service

Service เล็ก ๆ ที่ห่อ `faster-whisper` ไว้หลัง HTTP API เพื่อให้ Flask
webapp (โปรเจคแม่) เรียกใช้งานจาก localhost / โฮสต์จริง โดยไม่ต้อง
SSH เข้ามารัน docker บนเซิร์ฟเวอร์ทุกครั้ง

- Image base: `faster-whisper-gpu` (มีอยู่บน server ws1-rtx5090 แล้ว)
- Model: `large-v3` (cache อยู่ที่ `~/.cache/huggingface` บน server)
- Endpoint หลัก: `POST /transcribe` รับ multipart audio + คืน JSON
  ที่มี `segments[].words[]` (word-level timestamps) ใน schema เดียว
  กับ `sentence_recognition.SentenceRecognition.recognize`

โครงไฟล์:

```
server/
├── app.py              # FastAPI app
├── Dockerfile          # FROM faster-whisper-gpu + FastAPI deps
├── requirements.txt    # fastapi, uvicorn, python-multipart
└── README.md           # ไฟล์นี้
```

## Quick start (ฝั่ง server)

ดูคู่มือเต็มภาษาไทย พร้อมคำสั่งทุกขั้นตอน ที่
`docs/REMOTE_WHISPER_SETUP.md` ในรากของโปรเจค

## API summary

### `GET /health`

```json
{ "status": "ok", "model": "large-v3", "device": "cuda", "compute_type": "float16" }
```

ใช้ตรวจว่า model โหลดเสร็จแล้วหรือยัง (จะตอบ 503 + `status:"loading"`
ระหว่าง startup)

### `POST /transcribe`

- `Authorization: Bearer <WHISPER_API_KEY>` (ถ้าตั้งค่า env ไว้ ไม่งั้น
  ข้ามได้)
- multipart fields:
  - `file` (required) — ไฟล์ audio/video
  - `sensitivity` — `off` | `sensitive` | `ultra` (default `off`)
  - `language` — รหัสภาษา ISO เช่น `en` (default `en`)

Response:

```json
{
  "text": "...",
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.34,
      "text": " Hello world.",
      "words": [
        { "word": " Hello", "start": 0.00, "end": 0.42, "probability": 0.98 },
        { "word": " world.", "start": 0.42, "end": 0.91, "probability": 0.97 }
      ]
    }
  ]
}
```

Error codes:

- `401` — API key ไม่ถูกหรือไม่มี (เฉพาะเมื่อ container ตั้งค่า key ไว้)
- `413` — ไฟล์ใหญ่กว่า `WHISPER_MAX_UPLOAD_MB` (default 1024 MB)
- `503` — model ยังโหลดไม่เสร็จ
- `500` — model ตอนถอดเสียงล้มเหลว (ดูรายละเอียดใน `docker logs`)

## Environment variables

| ตัวแปร | ค่าเริ่มต้น | คำอธิบาย |
| --- | --- | --- |
| `WHISPER_MODEL` | `large-v3` | ขนาด/ชื่อโมเดล faster-whisper |
| `WHISPER_DEVICE` | `cuda` | `cuda` หรือ `cpu` |
| `WHISPER_COMPUTE_TYPE` | `float16` (cuda) / `int8` (cpu) | precision ของ ctranslate2 |
| `WHISPER_DEFAULT_LANGUAGE` | `en` | ใช้เมื่อ client ไม่ส่ง `language` มา |
| `WHISPER_MAX_UPLOAD_MB` | `1024` | ขนาดไฟล์สูงสุดต่อ request |
| `WHISPER_API_KEY` | `` (ว่าง) | ถ้าตั้งค่า → ทุก request ต้องส่ง `Authorization: Bearer <key>` |
| `LOG_LEVEL` | `INFO` | uvicorn / app log level |
