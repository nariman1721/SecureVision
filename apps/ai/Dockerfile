FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TORCH_HOME=/srv/models/torch
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PORT=10000

WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY apps/ai/requirements.txt ./requirements.txt

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

COPY apps/ai/ .

EXPOSE 10000

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
