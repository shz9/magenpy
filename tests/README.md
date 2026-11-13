# Tests and validation

Run the commands below from the repository root.

## Test organization

Tests are grouped by domain: `tests/genotype`, `tests/ld`, `tests/data`
(data loading, annotations, parsing, and harmonization), `tests/simulation`,
`tests/utils`, `tests/cli`, and `tests/benchmarks`. Test runners and manual
environment helpers live in `tests/scripts`.

Run the full Python suite with `python -m pytest tests`, or select a category,
for example `python -m pytest tests/genotype`. Run the CLI smoke tests from the
repository root with `bash tests/cli/test_cli.sh`.

## Native backend validation

The [backend parity suite](genotype/test_magenpy_vs_plink_backend.py) compares native
Magenpy results against PLINK 1.9/2.0 and independently encoded genotype fixtures.
It covers sample/variant filtering, cached frequencies, BED padding and split
offsets, scoring, GWAS with per-SNP missing-call exclusion or mean imputation,
LD, and LD scores. Run it locally with:

```sh
python -m pip install -e ".[test]"
python tools/install_plink.py --destination .venv/plink-bin
PATH="$PWD/.venv/plink-bin:$PATH" python tests/scripts/run_backend_checks.py
```

The installer downloads dated official builds (PLINK 1.9: December 11, 2023;
PLINK 2.0 alpha 5: January 5, 2024) on Linux x86-64 and macOS. Alternatively,
`make test-plink-docker` builds the native extensions and runs the suite in a
Linux amd64 container, including on ARM hosts with Docker emulation.

[Native backend validation](../.github/workflows/backend-validation.yml) runs on
every push and pull request and gates PyPI uploads. Missing PLINK executables,
missing native extensions, failed downloads/builds, empty runs, and skipped tests
fail this gate. Reports are retained as CI artifacts. Ordinary local pytest runs
can skip this module when PLINK is unavailable; the mandatory runner cannot.

## Performance regression checks

See [benchmarks/README.md](../benchmarks/README.md) for the continuous native
benchmark, local commands, timing/memory methodology, and historical results.
Benchmark infrastructure tests are in `tests/benchmarks`.

## Automatic formatting and linting

Install the pinned tools in a virtual environment, apply automatic fixes, then
check the result:

```sh
python -m pip install -r requirements-lint.txt
make fix
make lint
```

`make fix` applies safe Ruff fixes and Python formatting, normalizes Cython
string quotes, and formats tracked C/C++ sources and headers with clang-format.
Generated Cython C/C++ files are excluded. Individual commands are available as
`make fix-python`, `make fix-cython`, and `make fix-cpp`.

On pushes and same-repository pull requests, `.github/workflows/lint.yml`
commits these fixes back to the source branch using the GitHub Actions bot.
The workflow then checks the formatted commit with independent Python, Cython,
and C/C++ jobs. Fork and Dependabot pull requests run checks without pushing.
The repository must permit Actions to write to the source branch; protected
branch rules still apply. Automatic formatting commits use `GITHUB_TOKEN`, so
other push-triggered workflows are not automatically retriggered by those commits.

The check-only commands are `make lint`, `make lint-python`, `make lint-cython`,
and `make lint-cpp`. Lint issues without safe automatic fixes still require a
code change. Cython semantic checks and Cppcheck analysis remain part of these
checks alongside Python and C/C++ formatting verification.

