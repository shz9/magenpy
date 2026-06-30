#!/usr/bin/env bash

# Build and run the PLINK-vs-magenpy backend comparison tests in Docker.
#
# Usage:
#   bash tests/run_plink_backend_docker.sh
#
# Optional environment variables:
#   DOCKER_PLATFORM=linux/amd64
#   IMAGE_NAME=magenpy-plink-backend-tests
#   PYTEST_ARGS="-v -m plink tests/test_magenpy_vs_plink_backend.py"

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
IMAGE_NAME="${IMAGE_NAME:-magenpy-plink-backend-tests}"
PYTEST_ARGS="${PYTEST_ARGS:--v -m plink tests/test_magenpy_vs_plink_backend.py}"

cd "$PROJECT_ROOT"

docker build \
    --platform "$DOCKER_PLATFORM" \
    -f containers/plink-backend-tests.Dockerfile \
    -t "$IMAGE_NAME" \
    .

docker run --rm \
    --platform "$DOCKER_PLATFORM" \
    "$IMAGE_NAME" \
    python -m pytest $PYTEST_ARGS
