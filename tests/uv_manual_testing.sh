#!/usr/bin/env bash

# A script to test the package with different Python versions manually using uv.
# May be useful for sanity checks before pushing changes to the repository.
#
# Usage:
# $ bash tests/uv_manual_testing.sh
#
# Optional environment variables:
#   PYTHON_VERSIONS="3.10 3.11"  Override the default Python versions.
#   UV_TEST_ENV_DIR=".uv-test-envs"  Override where temporary virtualenvs live.
#   KEEP_UV_ENVS=1  Keep virtualenvs after each run for debugging.

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH." >&2
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
ENV_ROOT="${UV_TEST_ENV_DIR:-$PROJECT_ROOT/.uv-test-envs}"

echo "Running tests from: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Virtualenv root: $ENV_ROOT"

# Define Python versions. Override with:
#   PYTHON_VERSIONS="3.10 3.11" bash tests/uv_manual_testing.sh
if [[ -n "${PYTHON_VERSIONS:-}" ]]; then
    read -r -a python_versions <<< "$PYTHON_VERSIONS"
else
    python_versions=("3.10" "3.11" "3.12" "3.13" "3.14")
fi

mkdir -p "$ENV_ROOT"

cleanup_env() {
    local env_dir="$1"
    if [[ "${KEEP_UV_ENVS:-0}" != "1" ]]; then
        rm -rf "$env_dir"
    else
        echo "Keeping environment for debugging: $env_dir"
    fi
}

for version in "${python_versions[@]}"
do
    echo -e "\n\n================================== Testing python $version =====================================\n\n"

    env_dir="$ENV_ROOT/magenpy-$version"

    uv python install "$version"
    uv venv --clear --python "$version" "$env_dir"

    if [[ -x "$env_dir/bin/python" ]]; then
        python_bin="$env_dir/bin/python"
    else
        python_bin="$env_dir/Scripts/python.exe"
    fi

    "$python_bin" --version

    (
        cd "$PROJECT_ROOT"
        make clean
        uv pip install --python "$python_bin" --no-cache -e ".[test]"
        uv pip list --python "$python_bin"
        "$python_bin" -m pytest -v
        bash "$SCRIPT_DIR/test_cli.sh"
    )

    cleanup_env "$env_dir"

    echo -e "\n\n================================================================================================\n\n"
done
