"""Apply cython-lint's quote fixer without treating successful edits as errors."""

import subprocess

from cython_lint.string_fixer import fix_strings


def main():
    paths = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.pyx", "*.pxd", "*.pxi"]
        )
        .decode()
        .split("\0")
    )
    for path in filter(None, paths):
        if fix_strings(path, never=False):
            print(f"Normalized Cython quotes: {path}")


if __name__ == "__main__":
    main()
