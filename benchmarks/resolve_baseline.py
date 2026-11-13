"""Resolve a benchmark baseline, including commits removed by force pushes."""

import argparse
import subprocess
import sys


def resolve(ref):
    result = subprocess.run(
        [
            "git",
            "rev-parse",
            "--verify",
            "--quiet",
            "--end-of-options",
            f"{ref}^{{commit}}",
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def select_baseline(ref, allow_fallback=False):
    if ref == "0" * 40 and allow_fallback:
        ref = "HEAD^"
    sha = resolve(ref)
    if sha:
        return sha
    # Full-history checkout only fetches reachable history. A push's previous
    # tip can be absent after a force push, but still fetchable by object ID.
    fetched = subprocess.run(
        ["git", "fetch", "--no-tags", "--", "origin", ref],
        stdout=sys.stderr,
        stderr=sys.stderr,
    )
    if fetched.returncode == 0:
        sha = resolve("FETCH_HEAD")
        if sha:
            return sha
    if not allow_fallback:
        raise ValueError(
            f"Cannot resolve or fetch explicitly requested baseline: {ref!r}"
        )
    sha = resolve("HEAD^")
    if sha:
        print(
            "::warning::Automatic benchmark baseline unavailable; using HEAD^ instead.",
            file=sys.stderr,
        )
        return sha
    print(
        "::warning::No baseline or parent commit available; recording an initial "
        "benchmark without a regression comparison.",
        file=sys.stderr,
    )
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref")
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args()
    try:
        print(select_baseline(args.ref, args.allow_fallback))
    except ValueError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
