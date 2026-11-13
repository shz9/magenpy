"""Exercise baseline selection with real local Git remotes and rewritten history."""

from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/resolve_baseline.py"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def commit(repo, message):
    git(
        repo,
        "-c",
        "user.name=Benchmark test",
        "-c",
        "user.email=benchmark@example.com",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "--allow-empty",
        "-m",
        message,
    )
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repositories(tmp_path):
    remote, checkout = tmp_path / "remote", tmp_path / "checkout"
    remote.mkdir()
    git(remote, "init", "--initial-branch=old")
    old_sha = commit(remote, "Old history")
    git(remote, "checkout", "--orphan", "main")
    parent_sha = commit(remote, "New history")
    commit(remote, "Current revision")
    git(remote, "branch", "-D", "old")
    git(tmp_path, "clone", "--no-local", "--single-branch", str(remote), str(checkout))
    # Simulate checkout after a force push: the old commit is not in the clone.
    missing = subprocess.run(
        ["git", "-C", str(checkout), "cat-file", "-e", old_sha], capture_output=True
    )
    assert missing.returncode != 0
    return remote, checkout, old_sha, parent_sha


def select(checkout, ref, fallback=False):
    return subprocess.run(
        [sys.executable, str(SCRIPT), ref, *(["--allow-fallback"] if fallback else [])],
        cwd=checkout,
        capture_output=True,
        text=True,
    )


def test_local_baseline_needs_no_remote(repositories):
    _, checkout, _, parent_sha = repositories
    git(checkout, "remote", "remove", "origin")
    result = select(checkout, "HEAD^")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == parent_sha
    assert not result.stderr


def test_fetches_previous_tip_after_force_push(repositories):
    _, checkout, old_sha, _ = repositories
    result = select(checkout, old_sha)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == old_sha
    assert "::warning::" not in result.stderr


def test_fetches_manual_remote_branch(repositories):
    remote, checkout, old_sha, _ = repositories
    git(remote, "branch", "baseline-only", old_sha)
    result = select(checkout, "baseline-only")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == old_sha


def test_unavailable_automatic_baseline_uses_parent(repositories):
    _, checkout, _, parent_sha = repositories
    result = select(checkout, "f" * 40, fallback=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == parent_sha
    assert "::warning::" in result.stderr and "HEAD^" in result.stderr


def test_invalid_manual_baseline_fails_without_fallback(repositories):
    _, checkout, _, _ = repositories
    result = select(checkout, "missing-baseline")
    assert result.returncode != 0
    assert not result.stdout.strip()
    assert "explicitly requested baseline" in result.stderr


def test_new_branch_zero_sha_uses_parent(repositories):
    _, checkout, _, parent_sha = repositories
    result = select(checkout, "0" * 40, fallback=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == parent_sha


def test_initial_commit_records_without_comparison(tmp_path):
    git(tmp_path, "init", "--initial-branch=main")
    commit(tmp_path, "Initial commit")
    result = select(tmp_path, "0" * 40, fallback=True)
    assert result.returncode == 0, result.stderr
    assert not result.stdout.strip()
    assert "without a regression comparison" in result.stderr
