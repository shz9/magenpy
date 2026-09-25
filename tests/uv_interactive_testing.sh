#!/usr/bin/env bash

# Launch IPython with the current magenpy checkout installed in an isolated,
# temporary uv environment. The environment and uv cache are removed when
# IPython exits.
#
# Usage:
#   tests/uv_interactive_testing.sh [IPython arguments]
#
# Optional environment variables:
#   UV_INTERACTIVE_PYTHON=3.12  Select the Python interpreter/version.
#   MAGENPY_EXTRAS='[cloud]'    Install optional magenpy extras.

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH." >&2
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/magenpy-ipython.XXXXXX")"
VENV_DIR="$TEMP_ROOT/venv"
export UV_CACHE_DIR="$TEMP_ROOT/uv-cache"
export IPYTHONDIR="$TEMP_ROOT/ipython"

cleanup() {
    rm -rf "$TEMP_ROOT"
}
trap cleanup EXIT

PYTHON_REQUEST="${UV_INTERACTIVE_PYTHON:-python3}"
PACKAGE_SPEC="$PROJECT_ROOT${MAGENPY_EXTRAS:-}"

echo "Creating temporary environment with Python $PYTHON_REQUEST..."
uv venv --python "$PYTHON_REQUEST" "$VENV_DIR"

if [[ -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
else
    PYTHON_BIN="$VENV_DIR/Scripts/python.exe"
fi

echo "Installing the current checkout and IPython..."
uv pip install --python "$PYTHON_BIN" --no-cache -e "$PACKAGE_SPEC" ipython

echo "Launching IPython. The temporary environment will be removed on exit."
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m IPython "$@"
