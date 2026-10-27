# magenpy WASM releases

`.github/workflows/wasm.yml` builds magenpy independently for Python 3.13 and
Pyodide 0.28.3 (`pyodide_2025_0_wasm32`) using Emscripten 4.0.9. It compiles the
three Cython/C++ extensions with single-threaded kernels without BLAS/OpenMP,
and checks real imports, Zarr storage, and LD matrix operations in Chromium/Pyodide.
It never modifies the package source in place and does not depend on VIPRS.

## Triggers and outputs

- Pushes/merges to `main` or `master`, pull requests, and manual runs build and test.
- Published GitHub releases build the release tag and attach the tested assets.
- Manual `release_tag` builds that tag with build helpers from the selected workflow
  branch. Set `publish=true` to attach the results to that existing release; the
  default is artifact-only. This supports tags created before this workflow existed.

The `dist/wasm/` output contains:

- `magenpy-VERSION-cp313-cp313-pyodide_2025_0_wasm32.whl`
- `magenpy-VERSION-cp313-cp313-pyodide_2025_0_wasm32.whl.sha256`
- `magenpy-wasm.json` (version, wheel checksum, source commit/hash, ABI and build tools)
- `smoke-report.json` (browser test result; retained with Actions artifacts)

Actions artifacts expire after 30 days. Release assets remain attached to their
versioned release. These files are already covered by the repository's `dist/` and
`build/` ignore rules; do not commit them. Only the pinned compiler SDK and pip
packages are cached. No additional GitHub secret is needed: only the attachment
job receives `contents: write`. Native/PyPI workflows are unchanged.

Once this workflow and its helpers have been committed to the default branch,
backfill the current release with:

```sh
gh workflow run wasm.yml --repo shz9/magenpy --ref master -f release_tag=v0.2.3 -f publish=true
```

Adjust the branch/tag as appropriate. The existing tag is checked out separately;
it is never moved or rewritten. Rerunning replaces the named WASM assets only.
For immutable releases, attach assets to the draft *before* publication. The
published-release workflow preserves already-attached immutable assets and fails
if they are missing; a manual upload cannot modify an immutable published release.

## Build locally

Install uv and Git, then from the magenpy repository root:

```sh
uv run --no-project --python 3.13 --with-requirements tools/wasm/requirements-build.txt pyodide xbuildenv install 0.28.3
mkdir -p build/wasm
git clone --depth 1 --branch 4.0.9 https://github.com/emscripten-core/emsdk.git build/wasm/emsdk
build/wasm/emsdk/emsdk install 4.0.9
build/wasm/emsdk/emsdk activate 4.0.9
source build/wasm/emsdk/emsdk_env.sh
uv run --no-project --python 3.13 --with-requirements tools/wasm/requirements-build.txt python tools/wasm/build.py
uv run --no-project --with playwright python -m playwright install chromium
uv run --no-project --with playwright python tools/wasm/smoke.py
```

Skip cloning an existing SDK. `--prepare-only` stages without compiling.
`--source /path/to/checkout --release-tag v0.2.3` builds another checkout and verifies
its package version. For an extracted archive without `.git`, also provide
`--source-commit COMMIT` to record its provenance. Local builds include any local
source edits; release builds should use clean tagged checkouts.

## Consumption by VIPRS

VIPRS resolves the latest stable magenpy release at build time. It reuses a wheel
with the matching package version and Python/Pyodide ABI after checking its SHA-256,
internal metadata, and WASM extensions. The checksum comes from GitHub's asset digest
or the `.whl.sha256` sidecar. If that release has no matching wheel, VIPRS builds
from the exact release commit. Network or validation errors fail instead of being
silently treated as absent assets. The wheel is then served with VIPRS's static app;
browsers do not look up the latest release themselves.
