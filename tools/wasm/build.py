"""Build magenpy's own single-threaded Pyodide wheel, without modifying source files."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import zipfile
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
PYODIDE_VERSION = "0.28.3"
WHEEL_TAG = "cp313-cp313-pyodide_2025_0_wasm32"


def digest_tree(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def stage(source, build):
    name = "magenpy"
    target = build / "magenpy-source"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir()
    shutil.copytree(source / name, target / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.pyc", "*.cpp", "*.c"))
    for filename in ("LICENSE", "README.md"):
        shutil.copyfile(source / filename, target / filename)
    metadata = tomllib.loads((source / "pyproject.toml").read_text())["project"]
    version = metadata.get("version")
    if not version:
        tree = ast.parse((source / name / "_version.py").read_text())
        version = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets))
    dependencies = [dep for dep in metadata["dependencies"] if not dep.startswith("psutil")]
    extensions = []
    for pyx in sorted((target / name).rglob("*.pyx")):
        relative = pyx.relative_to(target)
        extensions.append((str(relative.with_suffix("")).replace("/", "."), str(relative)))
    setup = f"""from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
extensions = [Extension(module, [path], language='c++', include_dirs=[np.get_include()],
                       define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                       extra_compile_args=['-O3', '-std=c++17']) for module, path in {extensions!r}]
setup(name={name!r}, version={version!r}, packages=find_packages(),
      description='WebAssembly build of {name} for summary-statistics inference',
      license_files=['LICENSE'], install_requires={dependencies!r},
      package_data={{{name!r}: ['config/*.ini', 'data/*.txt', 'data/*.gz', '**/*.pxd', '**/*.hpp']}},
      ext_modules=cythonize(extensions, compiler_directives={{'language_level': 3, 'boundscheck': False,
          'wraparound': False, 'nonecheck': False, 'cdivision': True}}), zip_safe=False)
"""
    (target / "setup.py").write_text(setup)
    (target / "pyproject.toml").write_text('[build-system]\nrequires = ["setuptools==80.9.0", "wheel==0.45.1", "Cython==3.1.2", "numpy==2.2.5"]\nbuild-backend = "setuptools.build_meta"\n')
    return target, {"version": version, "source_sha256": digest_tree(target / name), "extensions": [x[0] for x in extensions]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--source-commit", help="Commit for an extracted source archive without .git")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/wasm")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/wasm")
    parser.add_argument("--release-tag", default="", help="Verify package version against this release tag")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 13):
        raise SystemExit("Use Python 3.13 for Pyodide 0.28.3.")
    args.build_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    target, info = stage(args.source.resolve(), args.build_dir.resolve())
    if args.release_tag and Version(args.release_tag.removeprefix("v")) != Version(info["version"]):
        raise SystemExit("The package version does not match the requested release tag.")
    if args.prepare_only:
        print(target)
        return
    subprocess.run(["pyodide", "build", str(target), "--outdir", str(args.output.resolve())], check=True)
    wheel = args.output / f"magenpy-{info['version']}-{WHEEL_TAG}.whl"
    with zipfile.ZipFile(wheel) as archive:
        modules = [name for name in archive.namelist() if name.endswith(".so")]
        expected = {name.replace(".", "/") + ".cpython-313-wasm32-emscripten.so" for name in info["extensions"]}
        if len(modules) != 3 or set(modules) != expected or any(archive.read(name)[:4] != b"\0asm" for name in modules):
            raise SystemExit("The wheel does not contain the expected three WASM extensions.")
    commit = args.source_commit
    if commit is None and (args.source / ".git").exists():
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source, text=True).strip()
    checksum = hashlib.sha256(wheel.read_bytes()).hexdigest()
    info.update({"pyodide_version": PYODIDE_VERSION, "wheel_tag": WHEEL_TAG,
                 "wheel": wheel.name, "sha256": checksum, "source_commit": commit,
                 "compiler": "Emscripten 4.0.9; single-threaded; C++ kernels without BLAS/OpenMP",
                 "build_tools": (Path(__file__).parent / "requirements-build.txt").read_text().splitlines()})
    (args.output / "magenpy-wasm.json").write_text(json.dumps(info, indent=2) + "\n")
    wheel.with_suffix(".whl.sha256").write_text(f"{checksum}  {wheel.name}\n")
    print(f"Built {wheel}")


if __name__ == "__main__":
    main()
