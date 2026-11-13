# `magenpy`: *M*odeling and *A*nalysis of (Statistical) *Gen*etics data in *py*thon

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/magenpy.svg)](https://pypi.python.org/pypi/magenpy/)
[![PyPI version fury.io](https://badge.fury.io/py/magenpy.svg)](https://pypi.python.org/pypi/magenpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


[![Linux CI](https://github.com/shz9/magenpy/actions/workflows/ci-linux.yml/badge.svg)](https://github.com/shz9/magenpy/actions/workflows/ci-linux.yml)
[![MacOS CI](https://github.com/shz9/magenpy/actions/workflows/ci-osx.yml/badge.svg)](https://github.com/shz9/magenpy/actions/workflows/ci-osx.yml)
[![Windows CI](https://github.com/shz9/magenpy/actions/workflows/ci-windows.yml/badge.svg)](https://github.com/shz9/magenpy/actions/workflows/ci-windows.yml)
[![Docs Build](https://github.com/shz9/magenpy/actions/workflows/ci-docs.yml/badge.svg)](https://github.com/shz9/magenpy/actions/workflows/ci-docs.yml)
[![Binary wheels](https://github.com/shz9/magenpy/actions/workflows/wheels.yml/badge.svg)](https://github.com/shz9/magenpy/actions/workflows/wheels.yml)


[![Downloads](https://static.pepy.tech/badge/magenpy)](https://pepy.tech/project/magenpy)
[![Downloads](https://static.pepy.tech/badge/magenpy/month)](https://pepy.tech/project/magenpy)

`magenpy` is a Python package for modeling and analyzing statistical genetics data. 
The package provides tools for:

* Reading and processing genotype data in `plink` BED format.
* Efficient LD matrix construction and storage in [Zarr](https://zarr.readthedocs.io/en/stable/index.html) array format.
* Data structures for harmonizing various GWAS data sources.
  * Includes parsers for commonly used GWAS summary statistics formats.
* Simulating polygenic traits (continuous and binary) using complex genetic architectures.
    * Multi-cohort simulation scenarios (beta)
    * Simulations incorporating functional annotations in the genetic architecture (beta)
* Interfaces for performing association testing on simulated and real phenotypes.
* Preliminary support for processing and integrating genomic annotations with other data sources.

### Helpful links

- [Documentation](https://shz9.github.io/magenpy/)
- [Sister package `viprs`](https://github.com/shz9/viprs)
- [Linkage disequilibrium (LD) browser](https://shz9.github.io/ld_browser/)
- [Citation / BibTeX records](./CITATION.md)
- [Report issues/bugs](https://github.com/shz9/magenpy/issues)

### Test organization

Tests are grouped by domain: `tests/genotype`, `tests/ld`, `tests/data`
(data loading, annotations, parsing, and harmonization), `tests/simulation`,
`tests/utils`, and `tests/cli`. Test runners and manual environment helpers live
in `tests/scripts`.

Run the full Python suite with `python -m pytest tests`, or select a category,
for example `python -m pytest tests/genotype`. Run the CLI smoke tests from the
repository root with `bash tests/cli/test_cli.sh`.

### Native backend validation

The [backend parity suite](tests/genotype/test_magenpy_vs_plink_backend.py) compares native
Magenpy results against PLINK 1.9/2.0 and independently encoded genotype fixtures.
It covers sample/variant filtering, cached frequencies, BED padding and split
offsets, scoring, GWAS, LD, and LD scores. Run it locally with:

```sh
python -m pip install -e ".[test]"
python tools/install_plink.py --destination .venv/plink-bin
PATH="$PWD/.venv/plink-bin:$PATH" python tests/scripts/run_backend_checks.py
```

The installer downloads dated official builds (PLINK 1.9: December 11, 2023;
PLINK 2.0 alpha 5: January 5, 2024) on Linux x86-64 and macOS. Alternatively,
`make test-plink-docker` builds the native extensions and runs the suite in a
Linux amd64 container, including on ARM hosts with Docker emulation.

[Native backend validation](.github/workflows/backend-validation.yml) runs on
every push and pull request and gates PyPI uploads. Missing PLINK executables,
missing native extensions, failed downloads/builds, empty runs, and skipped tests
fail this gate. Reports are retained as CI artifacts. Ordinary local pytest runs
can skip this module when PLINK is unavailable; the mandatory runner cannot.

### Automatic formatting and linting

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

### AI Usage
[![AI-DECLARATION: assist](https://img.shields.io/badge/䷼%20AI--DECLARATION-assist-fef9c3?labelColor=fef9c3)](https://ai-declaration.md)

This project is developed using AI-assisted technologies. Please review our [AI Declaration](AI-DECLARATION.md) for full transparency on automated pipelines.
