import json
import re
from pathlib import Path

from huggingface_hub import DatasetCard, ModelCard
from jsonschema import Draft202012Validator

from gpn import register_auto_classes

ROOT = Path(__file__).parents[1]
HUB_DIR = ROOT / "hub"
MANIFEST = json.loads((HUB_DIR / "manifest.json").read_text())
MANIFEST_SCHEMA = json.loads((HUB_DIR / "manifest.schema.json").read_text())
REPORT = json.loads((HUB_DIR / "audits" / "2026-08-19.json").read_text())
BASELINE = json.loads(
    (ROOT / "tests" / "fixtures" / "published_model_baseline.json").read_text()
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def test_manifest_conforms_to_its_json_schema() -> None:
    Draft202012Validator.check_schema(MANIFEST_SCHEMA)
    Draft202012Validator(
        MANIFEST_SCHEMA,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    ).validate(MANIFEST)


def test_supported_manifest_matches_scientific_baseline() -> None:
    records = MANIFEST["supported_models"]
    expected_families = {
        "gpn": "ss",
        "gpn_msa": "msa",
        "gpn_star": "star",
        "phylogpn": "phylo",
        "sorghum_expression": "ss",
    }
    assert [record["key"] for record in records] == [
        "gpn",
        "gpn_msa",
        "gpn_star",
        "phylogpn",
        "sorghum_expression",
    ]
    assert len({record["repo_id"] for record in records}) == len(records)

    for record in records:
        baseline = BASELINE["models"][record["key"]]
        assert record["family"] == expected_families[record["key"]]
        register_auto_classes(record["family"])
        assert record["repo_id"] == baseline["model_id"]
        assert record["revision"] == baseline["revision"]
        assert COMMIT_RE.fullmatch(record["revision"])
        assert record["fixture"].endswith(f"#models.{record['key']}")
        assert "config.json" in record["required_files"]
        assert "model.safetensors" in record["required_files"]


def test_support_and_download_boundaries_are_explicit() -> None:
    by_key = {record["key"]: record for record in MANIFEST["supported_models"]}
    assert by_key["gpn"]["support"] == "training_and_inference"
    assert by_key["gpn_star"]["support"] == "training_and_inference"
    assert by_key["gpn_msa"]["support"] == "deprecated_inference_only"
    assert by_key["phylogpn"]["support"] == "inference_only"
    assert by_key["sorghum_expression"]["support"] == "inference_only"

    data_by_key = {
        record["key"]: record for record in MANIFEST["required_data_artifacts"]
    }
    archive = data_by_key["multiz100way_fixture_source"]
    assert archive["maintenance"] == "provenance_only_do_not_download"
    assert archive["required_files"] == ["99.zarr.tar.gz"]


def test_dated_public_metadata_audit_passed() -> None:
    assert REPORT["passed"] is True
    assert REPORT["generated_at"].startswith("2026-08-19T")
    assert REPORT["mode"] == "metadata_only_no_weight_or_msa_downloads"

    observed = {record["key"]: record for record in REPORT["supported_models"]}
    assert set(observed) == {record["key"] for record in MANIFEST["supported_models"]}
    for approved in MANIFEST["supported_models"]:
        record = observed[approved["key"]]
        assert record["passed"] is True
        assert record["approved_revision"] == approved["revision"]
        assert record["current_main_revision"] == approved["revision"]
        assert all(record["checks"].values())

    inventory = REPORT["inventory"]
    assert len(inventory["collections"]) == 5
    assert not [model for model in inventory["models"] if model["private"]]
    assert not [dataset for dataset in inventory["datasets"] if dataset["private"]]
    for asset in inventory["models"] + inventory["datasets"]:
        assert COMMIT_RE.fullmatch(asset["sha"])

    inventoried_ids = {
        asset["repo_id"] for asset in inventory["models"] + inventory["datasets"]
    }
    for collection in inventory["collections"]:
        for item in collection["items"]:
            if item["repo_type"] in {"model", "dataset"}:
                assert item["repo_id"] in inventoried_ids

    collection_counts = {
        collection["slug"]: len(collection["items"])
        for collection in inventory["collections"]
    }
    assert collection_counts == {
        "songlab/gpn-653191edcb0270ed05ad2c3e": 4,
        "songlab/gpn-msa-65319280c93c85e11c803887": 11,
        "songlab/gpn-star-68c0c055acc2ee51d5c4f129": 32,
        "songlab/sorghum-gene-expression-prediction-68963dd31658bfb98c07ae1b": 2,
        "songlab/traitgym-6796d4fbb825d5b94e65d30f": 5,
    }
    assert set(collection_counts) == set(MANIFEST["collections"])

    observed_data = {
        record["key"]: record for record in REPORT["required_data_artifacts"]
    }
    assert set(observed_data) == {
        record["key"] for record in MANIFEST["required_data_artifacts"]
    }
    for approved in MANIFEST["required_data_artifacts"]:
        record = observed_data[approved["key"]]
        assert record["repo_id"] == approved["repo_id"]
        assert record["approved_revision"] == approved["revision"]
        assert record["current_main_revision"] == approved["revision"]
        assert record["maintenance"] == approved["maintenance"]
        assert record["required_files"] == approved["required_files"]
        assert record["passed"] is True
        assert all(record["checks"].values())


def test_report_records_known_documentation_and_portability_findings() -> None:
    observed = {record["key"]: record for record in REPORT["supported_models"]}
    assert "implicit_registration_example" in observed["gpn"]["card"]["findings"]
    assert "implicit_registration_example" in observed["gpn_msa"]["card"]["findings"]
    assert "empty_or_stub_card" in observed["gpn_star"]["card"]["findings"]
    assert observed["gpn_star"]["portability"]["status"] == (
        "package_fallback_required"
    )
    assert (
        "remote_code_presented_as_primary" in (observed["phylogpn"]["card"]["findings"])
    )
    assert "ambiguous_license" in observed["phylogpn"]["card"]["findings"]
    assert (
        "generated_placeholders" in (observed["sorghum_expression"]["card"]["findings"])
    )
    assert "missing_license" in observed["sorghum_expression"]["card"]["findings"]
    assert "misleading_pipeline_metadata" in (observed["gpn_msa"]["card"]["findings"])
    assert (
        "misleading_pipeline_metadata"
        in (observed["sorghum_expression"]["card"]["findings"])
    )

    serialized = json.dumps(REPORT)
    assert "/scratch/" not in serialized
    assert "/accounts/" not in serialized


def test_audit_script_is_bounded_and_never_snapshots() -> None:
    source = (HUB_DIR / "audit_hub.py").read_text()
    assert "MAX_METADATA_BYTES = 1024 * 1024" in source
    assert "HfApi(token=False)" in source
    assert "snapshot_download" not in source


def test_supported_card_proposals_use_explicit_pinned_autoclasses() -> None:
    proposal_root = HUB_DIR / "card-proposals" / "models"
    proposal_by_key = {
        "gpn": proposal_root / "songlab--gpn-brassicales" / "README.md",
        "gpn_msa": proposal_root / "songlab--gpn-msa-sapiens" / "README.md",
        "gpn_star": (proposal_root / "songlab--gpn-star-hg38-v100-200m" / "README.md"),
        "phylogpn": proposal_root / "songlab--PhyloGPN" / "README.md",
        "sorghum_expression": (
            proposal_root / "songlab--gpn-brassicales-gxa-sorghum-v1" / "README.md"
        ),
    }

    for record in MANIFEST["supported_models"]:
        source = proposal_by_key[record["key"]].read_text()
        assert record["repo_id"] in source
        assert record["revision"] in source
        assert f'register_auto_classes("{record["family"]}")' in source
        assert record["auto_class"] in source
        assert "revision=REVISION" in source
        assert "import gpn.model" not in source
        assert "trust_remote_code=True" not in source


def test_card_proposal_frontmatter_parses_offline() -> None:
    proposal_root = HUB_DIR / "card-proposals"
    model_cards = sorted((proposal_root / "models").glob("*/README.md"))
    dataset_cards = sorted((proposal_root / "datasets").glob("*/README.md"))
    assert len(model_cards) == 6
    assert len(dataset_cards) == 2
    for path in model_cards:
        source = path.read_text()
        assert ModelCard.load(path).data.to_dict()["inference"] is False
        registration_families = re.findall(
            r'register_auto_classes\("([^"]+)"\)', source
        )
        assert len(registration_families) == 1
        register_auto_classes(registration_families[0])
    for path in dataset_cards:
        DatasetCard.load(path)


def test_card_proposals_do_not_claim_dataset_workflow_support() -> None:
    proposal_text = "\n".join(
        path.read_text() for path in sorted((HUB_DIR / "card-proposals").rglob("*.md"))
    )
    normalized = re.sub(r"\s+", " ", proposal_text.replace(">", ""))
    assert "GPN-MSA training and dataset construction are no longer maintained" in (
        normalized
    )
    assert "does not maintain this dataset's construction workflow" in normalized
    assert "TODO(maintainer)" in normalized
