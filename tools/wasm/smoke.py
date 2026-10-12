"""Load the built magenpy wheel in real Pyodide and exercise its LD kernels."""
import argparse
import functools
import hashlib
import http.server
import json
import os
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=ROOT / "dist/wasm")
    args = parser.parse_args()
    info = json.loads((args.dist / "magenpy-wasm.json").read_text())
    assert hashlib.sha256((args.dist / info["wheel"]).read_bytes()).hexdigest() == info["sha256"]
    handler = functools.partial(QuietHandler, directory=str(args.dist.resolve()))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=os.environ.get("VIPRS_BROWSER_EXECUTABLE"), headless=True)
            page = browser.new_page()
            page.set_default_timeout(180000)
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            await_url = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/"
            page.add_script_tag(url=await_url + "pyodide.js")
            report = page.evaluate('''async ({info, indexURL}) => {
              const py = await loadPyodide({indexURL});
              await py.loadPackage(["micropip", "numpy", "scipy", "pandas", "joblib", "tqdm", "numcodecs", "asciitree", "statsmodels"]);
              py.globals.set("wheel_url", new URL(info.wheel, location.href).href + "#sha256=" + info.sha256);
              py.globals.set("expected_version", info.version);
              return JSON.parse(await py.runPythonAsync(`
import json, importlib, micropip
await micropip.install(["zarr==2.18.7", "fasteners==0.20", wheel_url], deps=False)
from tqdm import tqdm
tqdm.monitor_interval = 0
import magenpy
import numpy as np
from scipy.sparse import csr_matrix
from magenpy import LDMatrix
assert magenpy.__version__ == expected_version
modules = [importlib.import_module(name) for name in [
    "magenpy.stats.ld.c_utils", "magenpy.stats.score.score_cpp", "magenpy.stats.variant.variant_cpp"]]
assert all("wasm32-emscripten.so" in module.__file__ for module in modules)
matrix = 0.5 ** np.abs(np.arange(12)[:, None] - np.arange(12)[None, :])
ld = LDMatrix.from_csr(csr_matrix(matrix), "/tmp/smoke.zarr", dtype="float64", overwrite=True)
ld.set_store_attr("Chromosome", 22)
ld.set_store_attr("Sample size", 1000)
ld.set_metadata("snps", np.array([f"rs{i}" for i in range(12)]))
vector = np.arange(12, dtype="float64")
np.testing.assert_allclose(ld.dot(vector), matrix @ vector, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(ld.to_csr(return_symmetric=True).toarray(), matrix, rtol=1e-12, atol=1e-12)
json.dumps({"version": magenpy.__version__, "extensions": [m.__file__ for m in modules], "ld_roundtrip": True})
`));
            }''', {"info": info, "indexURL": await_url})
            (args.dist / "smoke-report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report))
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
