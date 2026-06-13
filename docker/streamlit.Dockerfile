# ============================================================================
# streamlit.Dockerfile — UI container
# ----------------------------------------------------------------------------
# This image only needs Streamlit + requests; we still install the full
# requirements.txt to keep environments aligned and keep iteration easy.
# ============================================================================
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt

COPY app ./app

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
