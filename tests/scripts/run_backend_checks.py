"""Run mandatory backend parity tests and reject empty or skipped test runs."""

import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    root = Path(__file__).resolve().parents[2]
    report = Path(
        os.environ.get("MAGENPY_BACKEND_REPORT", root / "backend-validation.xml")
    )
    env = dict(os.environ, MAGENPY_REQUIRE_PLINK="1")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            "-ra",
            "--strict-markers",
            f"--junitxml={report}",
            str(root / "tests/genotype/test_magenpy_vs_plink_backend.py"),
            *sys.argv[1:],
        ],
        cwd=root,
        env=env,
    )
    if result.returncode:
        return result.returncode
    cases = list(ET.parse(report).iter("testcase"))
    if not cases or any(case.find("skipped") is not None for case in cases):
        print(
            "Backend validation requires a nonempty run with ZERO skipped tests.",
            file=sys.stderr,
        )
        return 1
    print(f"Backend validation passed: {len(cases)} tests, zero skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
