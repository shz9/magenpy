# Build and run the PLINK-vs-magenpy backend comparison tests in a controlled
# Linux environment.
#
# Usage from the repository root:
#   docker build --platform linux/amd64 -f containers/plink-backend-tests.Dockerfile -t magenpy-plink-backend-tests .
#   docker run --rm --platform linux/amd64 magenpy-plink-backend-tests

FROM python:3.11-slim-bookworm

LABEL authors="Shadi Zabad"
LABEL description="Docker image for running magenpy PLINK backend comparison tests"

ARG PLINK2_URL="https://s3.amazonaws.com/plink2-assets/alpha5/plink2_linux_x86_64_20240105.zip"
ARG PLINK1_URL="https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20231211.zip"

ENV PATH="/software:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        gcc \
        libopenblas-dev \
        libomp-dev \
        make \
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

WORKDIR /workspace
COPY . /workspace

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -e ".[test]"

CMD ["python", "-m", "pytest", "-v", "-m", "plink", "tests/test_magenpy_vs_plink_backend.py"]
