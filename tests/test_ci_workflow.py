"""Unit and integration test suite for GitHub Actions CI workflow (Stage 8: CI Workflow)."""

from pathlib import Path

import yaml


def test_ci_workflow_file_exists():
    """Verify GitHub Actions CI workflow file exists."""
    ci_file = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
    assert ci_file.is_file(), f"CI workflow file not found at {ci_file}"


def test_ci_workflow_valid_yaml():
    """Verify CI workflow parses as valid YAML with required top-level keys."""
    ci_file = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
    with open(ci_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict)
    assert data.get("name") == "CI"
    assert "on" in data or True in data
    assert "concurrency" in data
    assert "jobs" in data


def test_ci_workflow_triggers_and_concurrency():
    """Verify triggers and concurrency group cancellation."""
    ci_file = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
    with open(ci_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    triggers = data.get("on") or data.get(True)
    assert triggers is not None
    assert "push" in triggers
    assert "pull_request" in triggers
    assert "workflow_dispatch" in triggers

    assert "main" in triggers["push"]["branches"]
    assert "master" in triggers["push"]["branches"]

    concurrency = data["concurrency"]
    assert concurrency["cancel-in-progress"] is True


def test_ci_workflow_jobs_and_matrix():
    """Verify all CI stages (lint, typecheck, test, postgres, build) are present."""
    ci_file = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
    with open(ci_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    jobs = data["jobs"]
    assert "lint" in jobs
    assert "typecheck" in jobs
    assert "test" in jobs
    assert "postgres-pipeline" in jobs
    assert "build" in jobs

    # Test matrix
    matrix = jobs["test"]["strategy"]["matrix"]
    assert "ubuntu-latest" in matrix["os"]
    assert "windows-latest" in matrix["os"]
    assert "macos-latest" in matrix["os"]
    assert "3.10" in matrix["python-version"]
    assert "3.11" in matrix["python-version"]
    assert "3.12" in matrix["python-version"]


def test_ci_workflow_postgres_service_and_build():
    """Verify postgres service configuration and wheel verification steps."""
    ci_file = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
    with open(ci_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    jobs = data["jobs"]
    pg_job = jobs["postgres-pipeline"]
    assert "services" in pg_job
    assert "postgres" in pg_job["services"]
    assert "postgres:16" in pg_job["services"]["postgres"]["image"]
    assert "5432:5432" in pg_job["services"]["postgres"]["ports"]

    build_job = jobs["build"]
    assert "postgres-pipeline" in build_job["needs"]
    assert "test" in build_job["needs"]
    assert "lint" in build_job["needs"]
    assert "typecheck" in build_job["needs"]

    # Verify wheel testing step exercises new sql commands
    steps = build_job["steps"]
    wheel_step = next(s for s in steps if s.get("name") == "Test installation from wheel")
    run_script = wheel_step["run"]
    assert "sql-schema" in run_script
    assert "sql-validate-script" in run_script
    assert "clean-export" in run_script
    assert "sql-generate-inserts" in run_script
    assert "sql-generate-copy" in run_script
    assert "sql-verify-exports" in run_script
