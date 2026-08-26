import json
import tomllib
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[1]
RELEASE_DIR = ROOT / "release"


def _json(name: str) -> dict:
    return json.loads((RELEASE_DIR / name).read_text())


def test_external_mutation_ledger_conforms_to_schema() -> None:
    schema = _json("external-mutations.schema.json")
    ledger = _json("external-mutations.json")

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(
        schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    ).validate(ledger)

    actions = ledger["applied"] + ledger["pending"]
    ids = [action["id"] for action in actions]
    assert len(ids) == len(set(ids))

    ready = [
        action
        for action in ledger["pending"]
        if action["disposition"] == "approval_ready"
    ]
    deferred = [
        action for action in ledger["pending"] if action["disposition"] == "deferred"
    ]
    assert ready
    assert {action["authorization"] for action in ready} == {
        "explicit_final_maintainer_approval"
    }
    if deferred:
        assert {action["authorization"] for action in deferred} == {
            "separate_future_maintainer_approval"
        }
    assert all(action["blocked_by"] for action in deferred)


def test_external_mutation_ledger_covers_release_boundary() -> None:
    ledger = _json("external-mutations.json")
    pending = {action["id"]: action for action in ledger["pending"]}
    applied = {action["id"]: action for action in ledger["applied"]}

    assert set(pending) == {
        "publish-gpn-0-9-0",
    }
    assert {
        "pypi-trusted-publisher-binding",
        "merge-modernization-pr",
        "publish-read-the-docs",
        "protect-main",
        "enable-security-features",
        "publish-analysis-archive",
    } <= set(applied)
    assert pending["publish-gpn-0-9-0"]["source"] == "release/0.9.0/review.md"
    assert pending["publish-gpn-0-9-0"]["disposition"] == "approval_ready"

    for action in ledger["pending"]:
        source = action.get("source")
        if source is not None:
            assert (ROOT / source).exists(), source


def test_external_mutation_dependencies_are_complete_and_ordered() -> None:
    ledger = _json("external-mutations.json")
    applied_ids = {action["id"] for action in ledger["applied"]}
    prior_ids = set(applied_ids)

    for action in ledger["pending"]:
        assert set(action.get("depends_on", [])) <= prior_ids
        prior_ids.add(action["id"])


def test_final_approval_has_an_exact_bounded_action_set() -> None:
    ledger = _json("external-mutations.json")
    ready_ids = {
        action["id"]
        for action in ledger["pending"]
        if action["disposition"] == "approval_ready"
    }

    assert ready_ids == {
        "publish-gpn-0-9-0",
    }


def test_hub_audit_is_outside_this_release() -> None:
    ledger = _json("external-mutations.json")
    assert all(action["system"] != "huggingface" for action in ledger["pending"])
    assert not (ROOT / "hub").exists()
    assert "issue #81" in (ROOT / "docs" / "development" / "release.md").read_text()


def test_archive_mutation_identifies_tag_object_and_peeled_commit() -> None:
    ledger = _json("external-mutations.json")
    applied = {action["id"]: action for action in ledger["applied"]}
    archive = applied["publish-analysis-archive"]

    assert "tag object 312a6c70de6700e729bcea4c9a67ab42a72f05f7" in archive["targets"]
    assert "commit 30dee6cf45849dfdcfc043ca8baf44fd6ba51d74" in archive["targets"]


def test_proposed_main_ruleset_matches_ci_and_solo_maintenance() -> None:
    ruleset = _json("main-ruleset.json")
    ledger = _json("external-mutations.json")
    assert ruleset["enforcement"] == "active"
    assert ruleset["bypass_actors"] == []
    assert ruleset["conditions"]["ref_name"] == {
        "include": ["~DEFAULT_BRANCH"],
        "exclude": [],
    }

    rules = {rule["type"]: rule for rule in ruleset["rules"]}
    assert {"deletion", "non_fast_forward", "pull_request"} <= set(rules)
    pull_request = rules["pull_request"]["parameters"]
    assert pull_request["required_approving_review_count"] == 0
    assert pull_request["require_code_owner_review"] is False
    assert pull_request["require_extra_approval_for_unattributed_changes"] is False
    assert pull_request["require_last_push_approval"] is False
    assert pull_request["required_review_thread_resolution"] is True
    assert pull_request["allowed_merge_methods"] == ["squash"]

    required_checks = rules["required_status_checks"]["parameters"][
        "required_status_checks"
    ]
    required = {item["context"] for item in required_checks}
    assert required == {
        "Quality",
        "Python 3.13",
        "Documentation",
        "Package",
    }
    protect_main = next(
        action for action in ledger["applied"] if action["id"] == "protect-main"
    )
    assert "songlab-cal/gpn:ruleset:21510261" in protect_main["targets"]
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    for context in required:
        assert f"name: {context}" in ci
    assert "matrix.python-version" not in ci
    assert {item["integration_id"] for item in required_checks} == {15368}
    assert (
        rules["required_status_checks"]["parameters"][
            "strict_required_status_checks_policy"
        ]
        is True
    )


def test_release_workflow_uses_locked_isolated_trusted_publishing() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()

    assert "permissions:\n  contents: read" in workflow
    assert "if: startsWith(github.event.release.tag_name, 'v')" in workflow
    assert "persist-credentials: false" in workflow
    assert "--only-group release --no-install-project" in workflow
    assert "uv build --no-build-isolation" in workflow
    assert "uvx" not in workflow
    assert "environment:\n      name: pypi" in workflow
    assert "permissions:\n      id-token: write" in workflow
    assert "attestations: true" in workflow
    assert 'test "v$(uv version --short)" = "${RELEASE_TAG}"' in workflow
    assert 'git merge-base --is-ancestor "${GITHUB_SHA}" origin/main' in workflow


def test_transformers_version_is_pinned_to_the_validated_runtime() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["requires-python"] == ">=3.13,<3.14"
    transformer_requirement = next(
        requirement
        for requirement in project["project"]["dependencies"]
        if requirement.startswith("transformers")
    )
    assert transformer_requirement == "transformers==5.15.0"

    lock = (ROOT / "uv.lock").read_text()
    assert '{ name = "transformers", specifier = "==5.15.0" }' in lock

    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "transformers-lower-bound" not in ci
    assert '--no-deps "transformers==' not in ci


def test_static_quality_checks_cover_the_full_package() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    mypy = project["tool"]["mypy"]
    assert mypy["files"] == ["src/gpn"]
    assert mypy["strict"] is True
    assert mypy["follow_imports"] == "skip"
    assert {
        "disallow_subclassing_any",
        "disallow_untyped_calls",
        "disallow_untyped_decorators",
        "warn_return_any",
    } == {name for name, value in mypy.items() if value is False}

    pre_commit = (ROOT / ".pre-commit-config.yaml").read_text()
    assert "- id: ruff-check\n" in pre_commit
    assert "- id: ruff-format\n" in pre_commit
    assert "name: mypy full package" in pre_commit
    assert "pass_filenames: false" in pre_commit
    assert "files: ^(pyproject\\.toml|src/gpn/.*\\.py)$" in pre_commit


def test_release_version_metadata_is_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    version = project["project"]["version"]
    locked_project = next(
        package
        for package in lock["package"]
        if package["name"] == "gpn" and package["source"] == {"editable": "."}
    )
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert version == "0.9.0"
    assert locked_project["version"] == version
    assert f"## {version} — 2026-08-25\n" in changelog


def test_release_review_records_the_modernization_boundary() -> None:
    review = (RELEASE_DIR / "0.9.0" / "review.md").read_text()
    runbook = (ROOT / "docs" / "development" / "release.md").read_text()
    release_readme = (RELEASE_DIR / "README.md").read_text()
    ledger = _json("external-mutations.json")
    merge = next(
        action
        for action in ledger["applied"]
        if action["id"] == "merge-modernization-pr"
    )

    assert "pull/100" in review
    assert "305c29a1db9bf327c7d2bc049b8800d8dc131fdb" in review
    assert "pull/96" not in review
    assert "pull/98" not in review
    assert "pull/99" not in review
    assert "component-prs.json" not in review
    assert not (RELEASE_DIR / "0.9.0" / "component-prs.json").exists()
    assert merge["operation"] == (
        "squash-merged the approved modernization pull request"
    )
    assert "https://github.com/songlab-cal/gpn/pull/100" in merge["targets"]
    assert any("matched the reviewed PR tree" in item for item in merge["evidence"])

    governance_text = "\n".join(
        (review, runbook, release_readme, json.dumps(ledger))
    ).lower()
    for stale_term in (
        "bottom-up",
        "component pull request",
        "cumulative pr",
        "modernization stack",
        "stack tip",
    ):
        assert stale_term not in governance_text
