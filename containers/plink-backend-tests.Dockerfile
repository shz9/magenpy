# Build and run the PLINK-vs-magenpy backend comparison tests in a controlled
# Linux environment.
#
# Usage from the repository root:
#   docker build --platform linux/amd64 -f containers/plink-backend-tests.Dockerfile -t magenpy-plink-backend-tests .
#   docker run --rm --platform linux/amd64 magenpy-plink-backend-tests

FROM python:3.11-slim-bookworm

LABEL authors="Shadi Zabad"
LABEL description="Docker image for running magenpy PLINK backend comparison tests"

ENV PATH="/software:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MAGENPY_REQUIRE_PLINK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        gcc \
        libopenblas-dev \
        libomp-dev \
        make \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY tools/install_plink.py /tmp/install_plink.py
RUN python /tmp/install_plink.py --destination /software

WORKDIR /workspace
COPY . /workspace

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -e ".[test]"

CMD ["python", "tests/scripts/run_backend_checks.py"]
