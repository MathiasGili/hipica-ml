# ============================================================================
# scheduler.Dockerfile — nightly Programa scrape job
# ----------------------------------------------------------------------------
# Tiny image: only requests + APScheduler. The actual heavy lifting
# (scrape, OCR, predict) lives inside the API container; this is a dumb
# HTTP client that wakes up once a day and calls /predict_program.
# ============================================================================
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=America/Montevideo

RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata \
 && rm -rf /var/lib/apt/lists/* \
 && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime

WORKDIR /app

RUN pip install --upgrade pip \
 && pip install requests==2.32.3 APScheduler==3.10.4

COPY scheduler /app/scheduler

CMD ["python", "-m", "scheduler.main"]
