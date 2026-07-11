# ============================================================================
# api.Dockerfile — FastAPI inference service
# ----------------------------------------------------------------------------
# CPU-only image: the API does inference, not training. Same Python and same
# requirements.txt as every other container so feature engineering matches
# byte-for-byte across environments (anti-skew guarantee).
# ============================================================================
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Runtime OS dependencies:
#   curl            healthcheck
#   tesseract-ocr   OCR for the Programa distance badges
#   libreoffice-calc  converts Crystal-Reports .xls → .xlsx so we can pull
#                   the embedded badge images out. The Tabulada loader uses
#                   xlrd and doesn't need it, but /predict_program (Programa
#                   parsing in src/ingestion/program.py) shells out to
#                   `libreoffice --headless --convert-to xlsx`. Adds ~400 MB
#                   but there is no lighter alternative.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl tesseract-ocr libreoffice-calc \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/requirements-api.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt

# Bring in code. Volumes in docker-compose will overlay these for live-reload
# during development.
COPY src ./src
COPY api ./api

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
