import json
import tomllib
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[1]
RELEASE_DIR = ROOT / "release"


def _json(name: str) -> dict:
    return json.loads((RELEASE_DIR / name).read_text())


def _release_json(name: str) -> dict:
    return json.loads((RELEASE_DIR / "0.9.0" / name).read_text())


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
    assert ready and deferred
    assert {action["authorization"] for action in ready} == {
        "explicit_final_maintainer_approval"
    }
    assert {action["authorization"] for action in deferred} == {
        "separate_future_maintainer_approval"
    }
    assert all(action["blocked_by"] for action in deferred)


def test_external_mutation_ledger_covers_release_boundary() -> None:
    ledger = _json("external-mutations.json")
    pending = {action["id"]: action for action in ledger["pending"]}
    applied = {action["id"]: action for action in ledger["applied"]}

    assert set(pending) == {
        "merge-modernization-stack",
        "protect-main",
        "enable-security-features",
        "publish-analysis-archive",
        "publish-gpn-0-9-0",
        "publish-read-the-docs",
    }
    assert "pypi-trusted-publisher-binding" in applied
    assert pending["protect-main"]["source"] == "release/main-ruleset.json"
    assert pending["publish-gpn-0-9-0"]["source"] == "release/0.9.0/review.md"
    assert pending["merge-modernization-stack"]["disposition"] == "approval_ready"
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
        "merge-modernization-stack",
        "protect-main",
        "enable-security-features",
        "publish-analysis-archive",
        "publish-gpn-0-9-0",
    }


def test_hub_audit_is_outside_this_release() -> None:
    ledger = _json("external-mutations.json")
    assert all(action["system"] != "huggingface" for action in ledger["pending"])
    assert not (ROOT / "hub").exists()
    assert "issue #81" in (ROOT / "docs" / "development" / "release.md").read_text()


def test_archive_mutation_identifies_tag_object_and_peeled_commit() -> None:
    ledger = _json("external-mutations.json")
    pending = {action["id"]: action for action in ledger["pending"]}
    archive = pending["publish-analysis-archive"]

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
        action for action in ledger["pending"] if action["id"] == "protect-main"
    )
    assert (
        "the final CI workflow has reported all four required contexts named in "
        "release/main-ruleset.json on a pull request" in protect_main["preconditions"]
    )
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
    assert f"## {version} — Unreleased\n" in changelog


def test_release_component_manifest_is_a_contiguous_immutable_stack() -> None:
    manifest = _release_json("component-prs.json")
    components = manifest["components"]

    assert manifest["release"] == "0.9.0"
    assert manifest["base"] == {
        "ref": "main",
        "commit": "690557d949309cf4f4234554888bb5421c49aede",
    }
    assert [component["number"] for component in components] == [
        *range(88, 97),
        98,
    ]
    assert components[0]["base_ref"] == "main"
    assert all(component["state"] == "merged" for component in components[:8])
    assert components[8]["base_ref"] == "main"
    assert all(component["state"] == "open" for component in components[8:])
    for previous, current in zip(components[8:], components[9:]):
        assert current["base_ref"] == previous["head_ref"]
    assert all(len(component["head_commit"]) == 40 for component in components)
    assert manifest["final_component"]["base_ref"] == components[-1]["head_ref"]
    assert manifest["final_component"]["head_ref"] == "codex/final-release-prep"
    assert manifest["final_component"]["number"] == 99
    assert manifest["cumulative_pull_request"] == 100
