# คู่มือเชื่อม Faster Whisper บน GPU Server (REST API)

เอกสารนี้สรุปทุกขั้นตอนเพื่อย้ายการประมวลผล Faster Whisper จากที่รันใน
Flask process โดยตรง ไปเรียกผ่าน HTTP API ของ container ที่รันค้างอยู่
บน GPU Server `ws1-rtx5090` (`10.80.39.41`)

สถาปัตยกรรมหลังเปลี่ยน:

```
[Browser]
    |
    v
[Flask app.py @ localhost:5000]
    |  (subprocess) python sentence_recognition.py <wav>
    v
[sentence_recognition.py]
    |  HTTP POST /transcribe (multipart wav)
    v
[FastAPI container @ 10.80.39.41:8000]   <-- จากโฟลเดอร์ server/
    |  faster-whisper (cuda, float16)
    v
[Whisper Large v3 model in VRAM]
```

JSON ที่ฝั่ง server ตอบกลับ มี schema เหมือน output ของ
`sentence_recognition.py` รุ่นเดิมเป๊ะ → `stress_highlight.py` และ
`app.py` ไม่ต้องแก้

## 0. ตรวจสอบสิ่งที่ต้องมีก่อน

- [ ] ได้รับ SSH credential ของ `ws1-rtx5090` แล้ว (จากแบบฟอร์มที่
      `https://forms.gle/x2oTRuvN9DKmwTop9`)
- [ ] เครื่อง dev (Windows / โน้ตบุ๊ก) อยู่ใน LAN/VPN ของคณะที่ ping
      `10.80.39.41` เจอ
- [ ] Docker image `faster-whisper-gpu` ติดตั้งบน server แล้ว (ตาม
      เอกสาร `faster-whisper-guide`) — ตรวจด้วย `docker images | grep faster-whisper-gpu`
- [ ] โมเดล `large-v3` cache อยู่ที่ `~/.cache/huggingface` บน server แล้ว

ตั้งตัวแปรชั่วคราวให้คำสั่งทั้งหมดข้างล่างใช้ร่วมกัน (เปลี่ยนตามจริง):

```powershell
# บนเครื่อง dev (PowerShell)
$env:GPU_USER = "prawit"             # username ที่ admin ส่งมาทางอีเมล
$env:GPU_HOST = "10.80.39.41"
$env:WHISPER_API_KEY = "<YOUR_WHISPER_API_KEY>"   # ขอ shared secret จาก admin / sysadmin
```

```bash
# บน server (bash) ตั้งหลังจาก SSH เข้าไปแล้ว
export WHISPER_API_KEY="<YOUR_WHISPER_API_KEY>"     # ค่าเดียวกับฝั่ง client
```

`WHISPER_API_KEY` เป็น shared secret กัน abuse — ทั้งสองฝั่งต้องตรงกัน
ถ้าเว้นว่างทั้งคู่ จะปิด auth (ใช้ตอนทดสอบบน localhost ได้)

---

## ขั้นตอนภาพรวม

1. ทดสอบ FastAPI service บนเครื่อง dev (CPU ก็ได้) → ยืนยันว่า output
   ตรงกับของเดิม
2. Push โฟลเดอร์ `server/` ขึ้น GPU server แล้ว build + run container
3. ชี้ Flask app ไปที่ `http://10.80.39.41:8000`
4. ทดสอบ end-to-end ผ่าน browser

---

## 1. ทดสอบ FastAPI บนเครื่อง dev ก่อน (optional แต่แนะนำ)

ขั้นนี้ช่วยตรวจว่า code ใน `server/app.py` ใช้งานได้จริง โดยยังไม่ต้อง
แตะ GPU server

### 1.1 รัน FastAPI ใน venv ของโปรเจค (CPU mode)

```powershell
# จากรากโปรเจค
.\venv\Scripts\Activate.ps1
pip install -r .\server\requirements.txt
pip install faster-whisper==1.0.3

# CPU + int8 พอสำหรับทดสอบ smoke test (ช้ามาก ใช้ไฟล์สั้น ๆ)
$env:WHISPER_DEVICE = "cpu"
$env:WHISPER_COMPUTE_TYPE = "int8"
$env:WHISPER_MODEL = "large-v3"          # หรือ "tiny" ถ้าอยากรอเร็ว
$env:WHISPER_API_KEY = ""                # ปิด auth ตอนทดสอบ
uvicorn app:app --host 0.0.0.0 --port 8000 --app-dir .\server
```

ดูว่ามีบรรทัด `Loading faster-whisper model 'large-v3' (device=cpu, compute_type=int8)...`
แล้วต่อด้วย `Model ready in ...` → service พร้อมใช้

### 1.2 เช็ก endpoint

เปิด terminal ใหม่:

```powershell
curl http://localhost:8000/health
# {"status":"ok","model":"large-v3",...}

curl.exe -X POST `
  -F "file=@.\path\to\test.wav" `
  -F "sensitivity=off" `
  -F "language=en" `
  http://localhost:8000/transcribe
```

ควรได้ JSON ที่มี `segments[].words[]`

### 1.3 ทดสอบผ่าน Flask (end-to-end บน localhost)

ใน terminal ที่จะรัน Flask:

```powershell
$env:WHISPER_BACKEND = "remote"
$env:WHISPER_API_URL = "http://localhost:8000"
$env:WHISPER_API_KEY = ""
$env:WHISPER_API_TIMEOUT = "600"
$env:WHISPER_LANGUAGE = "en"

python .\app.py
```

อัปโหลดไฟล์เสียงสั้น ๆ ผ่านหน้าเว็บ → ควรเห็น log ของ
`sentence_recognition` ว่า `POST http://localhost:8000/transcribe ...`
และ VTT ที่ได้ควรเหมือนตอนรัน local backend (ลองเทียบกันได้โดยสลับ
`WHISPER_BACKEND=local` แล้วต้องติดตั้ง `requirements-local.txt` ก่อน)

ถ้าตรงนี้ผ่าน → พร้อมไป deploy บน server

---

## 2. Deploy FastAPI บน GPU Server

### 2.1 Push โค้ด `server/` ขึ้น server

จากเครื่อง dev (รากโปรเจค):

```powershell
# สร้างโฟลเดอร์เป้าหมายและ copy ไฟล์ทั้งหมดด้วย scp
ssh "$($env:GPU_USER)@$($env:GPU_HOST)" "mkdir -p ~/stt-api"
scp .\server\app.py .\server\Dockerfile .\server\requirements.txt .\server\README.md `
    "$($env:GPU_USER)@$($env:GPU_HOST):~/stt-api/"
```

ถ้าโปรเจคอยู่ใน git อยู่แล้ว จะใช้ `rsync` หรือ `git clone` แทนก็ได้
สาระคือต้องมี 4 ไฟล์นั้นใน `~/stt-api/` บน server

### 2.2 SSH เข้า server แล้ว build image

```bash
ssh $GPU_USER@10.80.39.41
cd ~/stt-api
ls -lh                # ตรวจว่ามี app.py, Dockerfile, requirements.txt
docker build -t stt-api .
```

ขั้น build ใช้เวลาแค่ไม่กี่วินาที เพราะ base image
`faster-whisper-gpu` มีของหนัก ๆ พร้อมแล้ว เราแค่ติดตั้ง fastapi/uvicorn

ตรวจว่า image ขึ้นแล้ว:

```bash
docker images stt-api
```

### 2.3 รัน container แบบค้าง

```bash
# ลบ container เก่าถ้ามี (ครั้งแรกข้ามได้)
docker rm -f stt-api 2>/dev/null

docker run -d \
  --name stt-api \
  --restart unless-stopped \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e WHISPER_API_KEY="$WHISPER_API_KEY" \
  -e WHISPER_MODEL=large-v3 \
  -e WHISPER_DEVICE=cuda \
  -e WHISPER_COMPUTE_TYPE=float16 \
  -e WHISPER_MAX_UPLOAD_MB=1024 \
  -e WHISPER_DEFAULT_LANGUAGE=en \
  stt-api
```

อธิบาย flag สำคัญ:

- `--restart unless-stopped` — ให้ container ขึ้นเองหลัง server reboot
- `--gpus all` — เปิดให้ container เห็น RTX 5090
- `-p 8000:8000` — map port ออกมานอก container (host ฟัง 0.0.0.0:8000)
- `-v ~/.cache/huggingface:/root/.cache/huggingface` — ใช้ model cache
  เดิมที่โหลดไว้แล้ว ไม่ต้อง download ใหม่ (~3 GB)

### 2.4 ตรวจว่า model load สำเร็จ

```bash
docker logs -f stt-api
# รอจนเห็น:
#   [INFO] Loading faster-whisper model 'large-v3' (device=cuda, compute_type=float16)...
#   [INFO] Model ready in XX.Xs
#   [INFO] Application startup complete.
#   [INFO] Uvicorn running on http://0.0.0.0:8000
# Ctrl-C เพื่อออกจาก follow log (container ยังรันอยู่)
```

### 2.5 Smoke test จากบน server

```bash
# จาก server เอง (loopback ไม่ผ่าน firewall)
curl http://localhost:8000/health

# ทดสอบถอดเสียงด้วยไฟล์ตัวอย่าง
curl -X POST \
  -H "Authorization: Bearer $WHISPER_API_KEY" \
  -F "file=@$HOME/stt-work/test2.wav" \
  -F "sensitivity=off" \
  -F "language=en" \
  http://localhost:8000/transcribe | head -c 500
```

ถ้าเห็น JSON เริ่มต้นด้วย `{"text":"..."` ถือว่าใช้ได้

### 2.6 Smoke test จากเครื่อง dev (ผ่าน LAN/VPN)

จาก PowerShell บนเครื่อง dev:

```powershell
curl.exe http://10.80.39.41:8000/health

curl.exe -X POST `
  -H "Authorization: Bearer $env:WHISPER_API_KEY" `
  -F "file=@.C:\Users\vrx\Downloads\GAMENIGHT3.mp4" `
  -F "sensitivity=off" `
  http://10.80.39.41:8000/transcribe
```

ถ้า connection refused / timeout → ดู [ภาคแก้ปัญหา](#troubleshooting)

---

## 3. ชี้ Flask ไปที่ server จริง

หลัง container ทำงานบน server แล้ว เปลี่ยน env บน Flask host:

```powershell
# Windows / PowerShell
$env:WHISPER_BACKEND = "remote"
$env:WHISPER_API_URL = "http://10.80.39.41:8000"
$env:WHISPER_API_KEY = "<YOUR_WHISPER_API_KEY>"
$env:WHISPER_API_TIMEOUT = "600"
$env:WHISPER_LANGUAGE = "en"

python .\app.py
```

หรือใส่ใน `.env` แล้วโหลดผ่าน script ของคุณเอง (ตอนนี้ `app.py` ไม่ได้
auto-load `.env` ต้อง export ก่อนรันเอง — ดู `.env.example` เป็น
template)

อัปโหลดไฟล์ผ่านหน้าเว็บ → ในเทอร์มินัล Flask ควรเห็น:

```
[sentence_recognition] POST http://10.80.39.41:8000/transcribe (timeout=600s)
[sentence_recognition] Wrote .../uploads/<id>.json (NN segments, language=en)
```

ฝั่ง server (`docker logs stt-api`) ควรเห็น:

```
[xxxxxx] transcribe start file=... size=X.XXMB sensitivity=off lang=en
[xxxxxx] transcribe done segments=NN language=en elapsed=X.XXs
```

`elapsed` บน RTX 5090 ควรอยู่ในระดับวินาที (ไฟล์ ~5 นาที ≈ 10–20 s)

---

## 4. คำสั่งดูแลรักษาที่ใช้บ่อย

```bash
# ดู log
docker logs --tail 200 -f stt-api

# Restart (ใช้เมื่อค้างหรือ deploy โค้ดใหม่)
docker restart stt-api

# Deploy เวอร์ชันใหม่ของ app.py
cd ~/stt-api
# (scp app.py ใหม่ทับก่อน)
docker build -t stt-api .
docker rm -f stt-api
docker run -d --name stt-api --restart unless-stopped --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e WHISPER_API_KEY="$WHISPER_API_KEY" \
  stt-api

# หยุด/ปิด container ตอนเลิกใช้ (ตามนโยบาย GPU server)
docker stop stt-api
# หรือลบทิ้ง
docker rm -f stt-api

# ดูการใช้ GPU/RAM ของ container
docker stats stt-api --no-stream
nvidia-smi
```

---

## 5. ขั้นต่อไป: ขยับขึ้น public web

ตอนนี้ container รับ request จากใน LAN เท่านั้น (IP `10.x.x.x` =
private) เมื่อ webapp ของเราขึ้น hosting จริง (อยู่นอก LAN) ต้องคุยกับ
admin คณะให้เลือกหนึ่งใน:

1. **Reverse proxy ของคณะ** — ขอให้ map `https://stt.informatics.buu.ac.th/`
   → `http://10.80.39.41:8000` (มี TLS, มี rate limit) **แนะนำที่สุด**
2. **VPN site-to-site** จาก hosting ไปคณะ — webapp เรียก
   `http://10.80.39.41:8000` ตรง ๆ ผ่าน tunnel
3. **Outbound webhook / queue** — ถ้านโยบายไม่ให้เปิด port ออกข้างนอก
   เลย ต้องสลับ pattern เป็น webapp ส่งงานเข้า queue → worker บน server
   ดึงไปทำ → push ผลกลับ (ออกแบบเพิ่ม)

วิธีไหนก็ตาม ฝั่งโค้ด **ไม่ต้องแก้** — แค่เปลี่ยน
`WHISPER_API_URL` เป็น URL ปลายทางที่ admin ให้มา และ
`WHISPER_API_KEY` ตามที่ตกลง

ข้อควรระวังเรื่องนโยบาย:

- เอกสาร GPU server บอก **Inference 2–4 ชั่วโมงต่อ session** — เราจะรัน
  container ค้างยาวนานกว่านั้น ต้องแจ้ง admin ขอ exception หรือนัดเวลา
  คุย ก่อน deploy production
- ห้ามใช้บัญชีร่วมกัน — ถ้ามีหลายคนใน team ใช้ ให้ผู้ดูแล project เป็น
  เจ้าของ container เพียงคนเดียว
- ถ้าใช้ผ่าน reverse proxy ต้องตั้ง `client_max_body_size` ให้รองรับ
  ขนาดไฟล์ที่ต้องการ (default nginx มักแค่ 1 MB)

---

## 6. Troubleshooting

### `connection refused` หรือ `timeout` จากเครื่อง dev

- เช็ก firewall บน server: `sudo ufw status` (ถ้า ufw เปิดอยู่ต้อง
  `sudo ufw allow from 10.80.0.0/16 to any port 8000` หรือใช้ rule
  ตามที่ admin กำหนด)
- เช็กว่า container expose port จริง: `docker port stt-api`
  → ต้องเห็น `8000/tcp -> 0.0.0.0:8000`
- เช็กว่าเครื่อง dev เห็น server: `Test-NetConnection 10.80.39.41 -Port 8000`

### `401 Invalid API key`

- `WHISPER_API_KEY` ฝั่ง client (Flask) ต้องตรงกับฝั่ง container
- ดูค่าปัจจุบันที่ container ใช้: `docker exec stt-api env | grep WHISPER_API_KEY`

### `503 Model is still loading`

- โมเดล large-v3 ใช้เวลาโหลด ~30–60 วินาทีแรก รอแล้วลองใหม่
- ถ้านานเกินนั้น → `docker logs stt-api` มักเห็น out-of-memory หรือ
  CUDA error

### `500 Transcription failed: ...`

- ดู `docker logs stt-api` หา traceback
- มักเกิดจากไฟล์ที่ ffmpeg อ่านไม่ออก / codec แปลก → ลอง remux เป็น
  wav ก่อนส่ง

### Output ไม่เหมือนกับ local backend

- คาดได้ว่า `compute_type=float16` (server) vs `int8` (local CPU) จะ
  ทำให้ค่า `probability` ต่างกันเล็กน้อย ถือว่าปกติ
- จำนวน segment / word ควรใกล้เคียงกัน ถ้าหายเยอะให้เช็กว่าค่า
  `sensitivity` ส่งไปตรงกัน

### Flask ขึ้น `WHISPER_BACKEND=remote but WHISPER_API_URL is not set`

- ลืม export env ก่อนรัน Flask — เปิด terminal ใหม่จะหายไป ต้อง set ทุกครั้ง

### ต้อง rollback ใช้ local backend ฉุกเฉิน

```powershell
pip install -r .\requirements-local.txt
$env:WHISPER_BACKEND = "local"
python .\app.py
```

---

## 7. รายการไฟล์ที่ถูกแก้ / สร้างจากการ implement นี้

สร้างใหม่:

- `server/app.py` — FastAPI service
- `server/Dockerfile`
- `server/requirements.txt`
- `server/README.md`
- `requirements-local.txt`
- `.env.example`
- `docs/REMOTE_WHISPER_SETUP.md` (ไฟล์นี้)

แก้:

- `sentence_recognition.py` — แยก `_recognize_local` / `_recognize_remote`,
  default backend เป็น `remote`
- `requirements.txt` — ลบ `faster-whisper`, `psutil` (ย้ายไป
  `requirements-local.txt`), เพิ่ม `requests`

ไม่แตะ:

- `app.py`, `stress_highlight.py`, `audio_analyzer.py`, templates/, static/



Claude
PowerShell:
powershell# ตั้งตัวแปรก่อน (เปลี่ยน yourname เป็น username จริง)
$env:GPU_USER = "yourname"
$env:GPU_HOST = "10.80.39.41"

# สร้างโฟลเดอร์บน server
ssh "$($env:GPU_USER)@$($env:GPU_HOST)" "mkdir -p ~/stt-api"

# Copy ไฟล์ขึ้น
scp "C:\66910080\EnglishProject\Closed-Captioning-Project\server\app.py" `
    "C:\66910080\EnglishProject\Closed-Captioning-Project\server\Dockerfile" `
    "C:\66910080\EnglishProject\Closed-Captioning-Project\server\requirements.txt" `
    "C:\66910080\EnglishProject\Closed-Captioning-Project\server\README.md" `
    "$($env:GPU_USER)@$($env:GPU_HOST):~/stt-api/"
หรือถ้าอยากสั้นกว่านั้น ใช้ wildcard แทนได้เลย:
powershell
scp "C:\66910080\EnglishProject\Closed-Captioning-Project\server\*" `
    "$($env:GPU_USER)@$($env:GPU_HOST):~/stt-api/"
หลัง scp เสร็จ ลอง verify ที่ server ด้วย:
bash
ssh yourname@10.80.39.41 "ls -lh ~/stt-api/"
ควรเห็น app.py, Dockerfile, requirements.txt ครบ ถึงจะไป docker build ต่อได้ครับ

ตั้งตัวแปร ($env:GPU_USER = "yourname") คือแค่บอก PowerShell บนเครื่อง dev ว่า "จำชื่อนี้ไว้ใช้ในคำสั่งต่อๆ ไป" เพื่อไม่ต้องพิมพ์ซ้ำทุกครั้ง ยังอยู่บนเครื่องตัวเองอยู่เลย
SSH (ssh user@10.80.39.41) คือการ เดินทางเข้าไปในเครื่อง GPU Server จริงๆ หลังจากนี้ทุกคำสั่งที่พิมพ์จะรันบน server ไม่ใช่เครื่องตัวเอง

ลำดับที่ถูกต้องในเอกสารคือ:

บนเครื่อง dev — ตั้งตัวแปร → รัน scp อัปโหลดไฟล์ขึ้น server
SSH เข้า server — แล้วค่อยรัน docker build และ docker run บนนั้น

สองขั้นนี้ทำคนละที่กันครับ