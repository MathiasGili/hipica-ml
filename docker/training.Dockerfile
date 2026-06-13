# ============================================================================
# training.Dockerfile — GPU training container
# ----------------------------------------------------------------------------
# Built on top of the official NVIDIA CUDA base image so that XGBoost can use
# `tree_method='hist'` + `device='cuda'`. Requires the NVIDIA Container
# Toolkit on the host (see docker-compose.yml's `deploy.resources.reservations.devices`).
#
# Run from the repo root:
#   docker compose --profile training run --rm training \
#     python -m src.training.train --cache --register
# ============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        build-essential libpq-dev curl ca-certificates \
 && ln -sf /usr/bin/python3.10 /usr/bin/python \
 && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
 && python -m pip install -r /tmp/requirements.txt

COPY src ./src
COPY notebooks ./notebooks

# Default entry-point: launch a Jupyter kernel for interactive work.
# Override with `docker compose run training python -m src.training.train`.
EXPOSE 8888
CMD ["python", "-m", "src.training.train", "--cache"]
