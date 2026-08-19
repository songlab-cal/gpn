"""Run the deliberate, metadata-only Hugging Face compatibility audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from huggingface_hub import HfApi, hf_hub_url

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "hub" / "manifest.json"
MAX_METADATA_BYTES = 1024 * 1024


def _read_small_text(repo_id: str, filename: str, revision: str) -> str:
    """Read a small public text asset without creating a Hub snapshot."""

    request = Request(
        hf_hub_url(repo_id, filename, revision=revision),
        headers={"User-Agent": "gpn-hub-audit/1"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310
        if (length := response.headers.get("Content-Length")) is not None:
            if int(length) > MAX_METADATA_BYTES:
                raise ValueError(
                    f"Refusing metadata file larger than 1 MiB: {filename}"
                )
        payload = response.read(MAX_METADATA_BYTES + 1)
    if len(payload) > MAX_METADATA_BYTES:
        raise ValueError(f"Refusing metadata file larger than 1 MiB: {filename}")
    return payload.decode("utf-8")


def _card_dict(info: Any) -> dict[str, Any]:
    card_data = getattr(info, "card_data", None)
    return card_data.to_dict() if card_data is not None else {}


def _isoformat(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _model_summary(info: Any) -> dict[str, Any]:
    card = _card_dict(info)
    return {
        "repo_id": info.id,
        "sha": info.sha,
        "last_modified": _isoformat(info.last_modified),
        "private": info.private,
        "gated": info.gated,
        "library_name": info.library_name,
        "pipeline_tag": info.pipeline_tag,
        "license": card.get("license"),
        "tags": sorted(info.tags or []),
    }


def _dataset_summary(info: Any) -> dict[str, Any]:
    card = _card_dict(info)
    return {
        "repo_id": info.id,
        "sha": info.sha,
        "last_modified": _isoformat(info.last_modified),
        "private": info.private,
        "gated": info.gated,
        "license": card.get("license"),
        "tags": sorted(info.tags or []),
    }


def _classify_model(repo_id: str, supported_ids: set[str]) -> str:
    name = repo_id.split("/", maxsplit=1)[-1]
    if repo_id in supported_ids:
        return "supported"
    if name in {"PhyloGPN2", "deprecated-gpn-arabidopsis"}:
        return "published_related_not_supported"
    if name == "gpn-animal-promoter":
        return "published_research_not_supported"
    if name.startswith("tokenizer-dna-"):
        return "supporting_tokenizer"
    if name.startswith(("gpn-star-", "mlm-baseline-")):
        return "family_checkpoint_not_individually_validated"
    return "related_not_supported"


def _in_model_scope(repo_id: str, prefixes: list[str]) -> bool:
    name = repo_id.split("/", maxsplit=1)[-1]
    return any(name.startswith(prefix) for prefix in prefixes)


def _in_dataset_scope(
    repo_id: str,
    fragments: list[str],
    names: list[str],
) -> bool:
    name = repo_id.split("/", maxsplit=1)[-1]
    lowered = name.lower()
    return name in names or any(fragment.lower() in lowered for fragment in fragments)


def _audit_supported_model(api: HfApi, record: dict[str, Any]) -> dict[str, Any]:
    pinned = api.model_info(
        record["repo_id"],
        revision=record["revision"],
        files_metadata=True,
    )
    head = api.model_info(record["repo_id"])
    filenames = {sibling.rfilename for sibling in pinned.siblings or []}
    config = json.loads(
        _read_small_text(record["repo_id"], "config.json", record["revision"])
    )
    card_text = _read_small_text(record["repo_id"], "README.md", record["revision"])
    configured_path = config.get("phylo_dist_path")
    checks = {
        "immutable_revision_resolved": pinned.sha == record["revision"],
        "main_matches_approved_revision": head.sha == record["revision"],
        "required_files_present": set(record["required_files"]).issubset(filenames),
        "safetensors_present": "model.safetensors" in filenames,
        "model_type_matches": config.get("model_type") == record["model_type"],
        "architecture_matches": record["architecture"]
        in config.get("architectures", []),
    }
    portability_findings = []
    name_or_path = config.get("_name_or_path")
    if isinstance(name_or_path, str) and name_or_path.startswith("/"):
        portability_findings.append("inert_absolute_name_or_path")
    portability: dict[str, Any] = {
        "status": "self_contained",
        "findings": portability_findings,
    }
    if configured_path:
        bundled = {
            "phylo_dist/in_clade.npy",
            "phylo_dist/pairwise.npy",
        }.issubset(filenames)
        portability = {
            "configured_path_kind": (
                "absolute" if str(configured_path).startswith("/") else "relative"
            ),
            "bundled_fallback_present": bundled,
            "status": "package_fallback_required" if bundled else "not_portable",
            "findings": portability_findings + ["nonportable_configured_path"],
        }
        checks["portable_auxiliary_assets"] = bundled
    card_findings = []
    if len(card_text.strip()) < 200:
        card_findings.append("empty_or_stub_card")
    if "[More Information Needed]" in card_text:
        card_findings.append("generated_placeholders")
    if "import gpn.model" in card_text:
        card_findings.append("implicit_registration_example")
    if "trust_remote_code=True" in card_text:
        card_findings.append("remote_code_presented_as_primary")
    license_name = _card_dict(pinned).get("license")
    if not license_name:
        card_findings.append("missing_license")
    elif license_name == "cc":
        card_findings.append("ambiguous_license")
    if pinned.library_name is None:
        card_findings.append("missing_library_metadata")
    pipeline_policy = record["hub_pipeline_policy"]
    if pipeline_policy == "local_fill_mask_only":
        if pinned.pipeline_tag != "fill-mask":
            card_findings.append("unexpected_pipeline_metadata")
    elif pinned.pipeline_tag is not None:
        card_findings.append("misleading_pipeline_metadata")
    return {
        "key": record["key"],
        "repo_id": record["repo_id"],
        "approved_revision": record["revision"],
        "current_main_revision": head.sha,
        "support": record["support"],
        "files": sorted(filenames),
        "config": {
            "architectures": config.get("architectures"),
            "auto_map": config.get("auto_map"),
            "model_type": config.get("model_type"),
            "name_or_path_kind": (
                "absolute"
                if isinstance(name_or_path, str) and name_or_path.startswith("/")
                else "relative"
                if isinstance(name_or_path, str)
                else None
            ),
        },
        "portability": portability,
        "card": {
            "license": license_name,
            "library_name": pinned.library_name,
            "pipeline_tag": pinned.pipeline_tag,
            "pipeline_policy": pipeline_policy,
            "findings": card_findings,
            "needs_update": bool(card_findings),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _audit_data_artifact(api: HfApi, record: dict[str, Any]) -> dict[str, Any]:
    pinned = api.dataset_info(
        record["repo_id"],
        revision=record["revision"],
        files_metadata=True,
    )
    head = api.dataset_info(record["repo_id"])
    filenames = {sibling.rfilename for sibling in pinned.siblings or []}
    checks = {
        "immutable_revision_resolved": pinned.sha == record["revision"],
        "main_matches_approved_revision": head.sha == record["revision"],
        "required_files_present": set(record["required_files"]).issubset(filenames),
    }
    return {
        "key": record["key"],
        "repo_id": record["repo_id"],
        "approved_revision": record["revision"],
        "current_main_revision": head.sha,
        "maintenance": record["maintenance"],
        "required_files": record["required_files"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_audit(manifest: dict[str, Any], api: HfApi | None = None) -> dict[str, Any]:
    """Query public metadata and return a serializable compatibility report.

    The default client deliberately ignores local credentials so a committed report
    cannot disclose private experiment names or revisions.
    """

    api = api or HfApi(token=False)
    supported = manifest["supported_models"]
    supported_ids = {record["repo_id"] for record in supported}
    scope = manifest["inventory_scope"]

    models = [
        info
        for info in api.list_models(
            author="songlab",
            limit=None,
            expand=[
                "cardData",
                "gated",
                "lastModified",
                "library_name",
                "pipeline_tag",
                "private",
                "sha",
                "tags",
            ],
        )
        if not info.private and _in_model_scope(info.id, scope["model_name_prefixes"])
    ]
    datasets = [
        info
        for info in api.list_datasets(
            author="songlab",
            limit=None,
            expand=[
                "cardData",
                "gated",
                "lastModified",
                "private",
                "sha",
                "tags",
            ],
        )
        if not info.private
        and _in_dataset_scope(
            info.id,
            scope["dataset_name_fragments"],
            scope["dataset_names"],
        )
    ]
    collections = {
        slug: api.get_collection(slug, token=False) for slug in manifest["collections"]
    }

    models_by_id = {info.id: info for info in models}
    datasets_by_id = {info.id: info for info in datasets}
    for collection in collections.values():
        for item in collection.items:
            if item.item_type == "model" and item.item_id not in models_by_id:
                info = api.model_info(item.item_id)
                if not info.private:
                    models_by_id[info.id] = info
            elif item.item_type == "dataset" and item.item_id not in datasets_by_id:
                info = api.dataset_info(item.item_id)
                if not info.private:
                    datasets_by_id[info.id] = info

    models = list(models_by_id.values())
    datasets = list(datasets_by_id.values())

    supported_audit = [_audit_supported_model(api, record) for record in supported]
    data_audit = [
        _audit_data_artifact(api, record)
        for record in manifest["required_data_artifacts"]
    ]
    report = {
        "schema_version": manifest["schema_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "metadata_only_no_weight_or_msa_downloads",
        "supported_models": supported_audit,
        "required_data_artifacts": data_audit,
        "inventory": {
            "models": [
                {
                    **_model_summary(info),
                    "classification": _classify_model(info.id, supported_ids),
                }
                for info in sorted(models, key=lambda item: item.id.lower())
            ],
            "datasets": [
                _dataset_summary(info)
                for info in sorted(datasets, key=lambda item: item.id.lower())
            ],
            "collections": [
                {
                    "slug": slug,
                    "title": collection.title,
                    "last_updated": _isoformat(collection.last_updated),
                    "description": collection.description,
                    "items": [
                        {
                            "repo_id": item.item_id,
                            "repo_type": item.item_type,
                            "note": item.note,
                            "position": item.position,
                        }
                        for item in collection.items
                    ],
                }
                for slug, collection in sorted(collections.items())
            ],
            "external_assets": scope["external_assets"],
        },
    }
    report["passed"] = all(item["passed"] for item in supported_audit + data_audit)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit GPN Hugging Face metadata. This never downloads model weights "
            "or alignment archives."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    output = json.dumps(run_audit(manifest), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output)


if __name__ == "__main__":
    main()
