"""Install the dated official PLINK builds used by the backend regression gate."""

import argparse
import io
import platform
from pathlib import Path
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile


BASE = "https://s3.amazonaws.com"


def install(destination):
    system, machine = platform.system(), platform.machine().lower()
    if system == "Linux" and machine in {"x86_64", "amd64"}:
        plink1, plink2 = "linux_x86_64", "linux_x86_64"
    elif system == "Darwin":
        plink1 = "mac"
        plink2 = "mac_arm64" if machine == "arm64" else "mac"
    else:
        raise RuntimeError(
            "Use the linux/amd64 PLINK test Docker image on this platform."
        )
    urls = {
        "plink": f"{BASE}/plink1-assets/plink_{plink1}_20231211.zip",
        "plink2": f"{BASE}/plink2-assets/alpha5/plink2_{plink2}_20240105.zip",
    }
    destination.mkdir(parents=True, exist_ok=True)
    for executable, url in urls.items():
        for attempt in range(3):
            try:
                with urlopen(url, timeout=120) as response:
                    archive_bytes = response.read()
                break
            except URLError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        # Read only the named binary rather than extracting arbitrary zip paths.
        with ZipFile(io.BytesIO(archive_bytes)) as archive:
            binary = destination / executable
            binary.write_bytes(archive.read(executable))
        binary.chmod(0o755)
        subprocess.run([str(binary.resolve()), "--version"], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    install(parser.parse_args().destination)
