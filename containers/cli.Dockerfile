# Docker image for the standard magenpy command-line tools.
#
# Build from the latest PyPI release:
#   docker build --platform linux/amd64 -f containers/cli.Dockerfile -t magenpy-cli .
#
# Build from this repository checkout:
#   docker build --platform linux/amd64 -f containers/cli.Dockerfile --build-arg MAGENPY_INSTALL_TARGET=. -t magenpy-cli .
#
# Run a command:
#   docker run --rm magenpy-cli magenpy_ld -h
#   docker run --rm magenpy-cli magenpy_simulate -h

FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.authors="Shadi Zabad" \
      org.opencontainers.image.title="magenpy CLI" \
      org.opencontainers.image.description="Docker image for the standard magenpy command-line tools" \
      org.opencontainers.image.source="https://github.com/shz9/magenpy" \
      org.opencontainers.image.licenses="MIT"

ARG MAGENPY_INSTALL_TARGET="magenpy"
ARG PLINK2_URL="https://s3.amazonaws.com/plink2-assets/alpha5/plink2_linux_x86_64_20240105.zip"
ARG PLINK1_URL="https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20231211.zip"

ENV PATH="/software:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        gcc \
        libopenblas-dev \
        libomp-dev \
        pkg-config \
        unzip \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /software \
    && wget -q "${PLINK2_URL}" -O /software/plink2.zip \
    && unzip -q /software/plink2.zip -d /software \
    && rm /software/plink2.zip \
    && wget -q "${PLINK1_URL}" -O /software/plink.zip \
    && unzip -q /software/plink.zip -d /software \
    && rm /software/plink.zip \
    && chmod +x /software/plink /software/plink2 \
    && plink2 --version \
    && plink --version

WORKDIR /opt/magenpy
COPY . /opt/magenpy

RUN python -m pip install --upgrade pip \
    && python -m pip install "${MAGENPY_INSTALL_TARGET}" \
    && magenpy_ld -h >/dev/null \
    && magenpy_simulate -h >/dev/null

WORKDIR /work

CMD ["magenpy_ld", "-h"]
