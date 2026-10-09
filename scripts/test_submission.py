"""Regression tests for relocated, authenticated-by-hash evidence and paired analysis."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import analyze_results
import verify


def test_paired_accuracy_requires_the_same_questions():
    with pytest.raises(ValueError, match="match"):
        analyze_results.paired_accuracy([{"prompt": "a", "answer": "1", "correct": True}],
                                        [{"prompt": "b", "answer": "1", "correct": True}])


def test_unchanged_paired_predictions_have_zero_difference():
    rows = [{"prompt": str(i), "answer": "1", "correct": bool(i % 2)} for i in range(10)]
    result = analyze_results.paired_accuracy(rows, rows)
    assert result["delta"] == 0
    assert result["paired_bootstrap_ci95"] == [0, 0]
    assert result["mcnemar_exact_p"] == 1


def test_relocated_reference_requires_unchanged_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "REPO", tmp_path)
    adapter = tmp_path / "adapters/dpo"
    adapter.mkdir(parents=True)
    folder = tmp_path / "submission/provenance"
    folder.mkdir(parents=True)
    origin = "/kaggle/working/lab22-v2/models/sft-merged"
    config = adapter / "adapter_config.json"
    config.write_text(json.dumps({"base_model_name_or_path": origin}))
    snapshot = folder / "sft-model-config.json"
    snapshot.write_text('{"model_type":"qwen3"}')
    hashes = {"adapters/dpo/adapter_config.json": {"sha256": hashlib.sha256(config.read_bytes()).hexdigest()},
              "models/sft-merged/config.json": {"sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}}
    (folder / "import.json").write_text(json.dumps({"original_reference": origin, "file_hashes": hashes}))
    assert verify.verified_import_reference(origin, adapter)
    config.write_text(json.dumps({"base_model_name_or_path": origin, "tampered": True}))
    assert not verify.verified_import_reference(origin, adapter)
