"""Import verified evidence from the return ZIP without executing or overwriting its source."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parent.parent


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected(name: str) -> bool:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or any(part.startswith(".") for part in p.parts):
        return False
    if p.parts[0] == "notebooks":
        return p.suffix == ".ipynb"
    if p.parts[0] == "adapters":
        return p.name in {"adapter_config.json", "dpo_metrics.json", "split.json", "variants_summary.json", "grpo_metrics.json"}
    if p.parts[0] == "models":
        return name == "models/sft-merged/config.json"
    if p.parts[0] == "data":
        return p.suffix in {".json", ".jsonl", ".parquet"}
    if p.parts[0] == "submission":
        return p.name != "REFLECTION.md" and p.suffix in {".json", ".log", ".txt", ".png", ".jpg"}
    return False


def import_results(archive_path: Path, destination: Path = REPO) -> dict:
    secret = re.compile(rb"(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{25,}|hf_[A-Za-z0-9]{25,})")
    imported, source_checks = {}, {}
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("RESULTS_MANIFEST.json"))
        for name, meta in manifest.items():
            body = archive.read(name)
            if digest(body) != meta["sha256"] or len(body) != meta["bytes"]:
                raise ValueError(f"Manifest mismatch: {name}")
            path = destination / name
            if name.endswith(".py") and path.exists():
                source_checks[name] = digest(path.read_bytes()) == meta["sha256"]
            if not selected(name):
                continue
            if path.suffix in {".json", ".jsonl", ".log", ".txt", ".ipynb"} and secret.search(body):
                raise ValueError(f"Possible secret in {name}; redact before publishing")
            if path.exists() and path.read_bytes() != body:
                raise ValueError(f"Evidence differs from existing file: {name}; preserve both runs separately")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            imported[name] = meta
        adapter = json.loads(archive.read("adapters/dpo/adapter_config.json"))
    provenance = {
        "archive_name": archive_path.name,
        "archive_sha256": digest(archive_path.read_bytes()),
        "original_reference": adapter["base_model_name_or_path"],
        "file_hashes": imported,
        "source_matches": source_checks,
        "weights_in_archive": False,
        "note": "Imported evidence remains byte-for-byte unchanged. Source and REFLECTION are maintained in the working repo.",
    }
    folder = destination / "submission/provenance"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "import.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    (folder / "kaggle-manifest.json").write_text(json.dumps(manifest, indent=2))
    (folder / "sft-model-config.json").write_bytes((destination / "models/sft-merged/config.json").read_bytes())
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    result = import_results(args.archive)
    print(f"Imported {len(result['file_hashes'])} evidence files; archived sources were not executed.")
