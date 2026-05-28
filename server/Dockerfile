# Faster Whisper FastAPI service.
#
# Builds on top of the pre-built ``faster-whisper-gpu`` image that already
# exists on the GPU server (ws1-rtx5090). That image includes Python,
# CUDA, cuDNN, ctranslate2 and faster-whisper. We only need to add the
# FastAPI HTTP surface and our app.py.
FROM faster-whisper-gpu

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py

# NOTE: WHISPER_API_KEY is INTENTIONALLY not baked into the image.
# Pass it at runtime so the secret never ends up in the image layers:
#   docker run -e WHISPER_API_KEY="<shared-secret>" ... closed-captioning-whisper
# An empty value disables auth (useful only for localhost smoke tests).
ENV WHISPER_MODEL=large-v3 \
    WHISPER_DEVICE=cuda \
    WHISPER_COMPUTE_TYPE=float16 \
    WHISPER_DEFAULT_LANGUAGE=en \
    WHISPER_MAX_UPLOAD_MB=1024 \
    WHISPER_API_KEY="" \
    LOG_LEVEL=INFO

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
