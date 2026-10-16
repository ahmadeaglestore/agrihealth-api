# AgriHealth AI backend – HALKA version (ONNX, PyTorch ke baghair)
# Render, Koyeb ya kisi bhi Docker server par chalta hai. 512 MB RAM kaafi hai.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TORCH_THREADS=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user . .
USER user

# Server khud PORT batata hai (Render: 10000). Na bataye to 7860.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
