"""CPU contract tests for the Kaggle deliverable and evidence handling."""
from __future__ import annotations

import ast
import base64
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import build_kaggle
import kaggle_runner as runner
from kaggle_setup import ensure_pip


def test_setup_creates_venv_when_session_tmp_directory_is_gone(tmp_path):
    import subprocess

    venv = tmp_path / "new-session" / "venv"
    assert not (venv / "bin/python").exists()
    python = ensure_pip(venv)
    assert Path(python).is_file()
    result = subprocess.run([python, "-I", "-m", "pip", "--version"], capture_output=True, text=True)
    assert result.returncode == 0


def test_uploaded_setup_cell_creates_interpreter_before_pip_probe(tmp_path):
    import os
    import subprocess

    nb = build_kaggle.render()
    setup = next("".join(c["source"]) for c in nb["cells"]
                 if c["cell_type"] == "code" and "PY = ensure_pip(VENV)" in "".join(c["source"]))
    prefix = setup.split("requirements =", 1)[0]
    namespace = {"ROOT": REPO, "VENV": tmp_path / "fresh-session" / "venv",
                 "NOTEBOOK_VERSION": "test", "os": os}
    exec(compile(prefix, "uploaded-cell-D", "exec"), namespace)
    python = namespace["PY"]
    assert Path(python).exists()
    assert subprocess.run([python, "-I", "-m", "pip", "--version"], capture_output=True).returncode == 0


def test_configured_repair_notebook_preserves_user_provider_and_uses_fresh_paths():
    nb = build_kaggle.render({"ROOT": "/kaggle/working/lab22-v2", "VENV": "/tmp/lab22-kaggle-venv-v2",
                             "API_JUDGE_PROVIDER": "openai", "API_JUDGE_MODEL": "gpt-4.1-mini",
                             "HF_REPO": "duckak12/lab22-dpo-vn-experimental", "PUBLISH_HF": True})
    source = "".join(nb["cells"][2]["source"])
    tree = ast.parse(source)
    constants = {s.targets[0].id: ast.literal_eval(s.value) for s in tree.body
                 if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) and isinstance(s.value, ast.Constant)}
    assert constants["API_JUDGE_PROVIDER"] == "openai"
    assert constants["API_JUDGE_MODEL"] == "gpt-4.1-mini"
    assert constants["PUBLISH_HF"] is True
    assert "lab22-kaggle-venv-v2" in source


def test_setup_repairs_existing_venv_without_pip_and_ignores_host_hook(tmp_path, monkeypatch):
    import subprocess

    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-I", "-m", "venv", "--without-pip", str(venv)], check=True)
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text("import wrapt\n")
    monkeypatch.setenv("PYTHONPATH", str(hook))
    broken = subprocess.run([str(venv / "bin/python"), "-m", "pip", "--version"], capture_output=True, text=True)
    assert broken.returncode != 0 and "No module named pip" in broken.stderr
    assert "No module named 'wrapt'" in broken.stderr
    python = ensure_pip(venv)
    repaired = subprocess.run([python, "-I", "-m", "pip", "--version"], capture_output=True, text=True)
    assert repaired.returncode == 0
    assert "sitecustomize" not in repaired.stderr


def test_upload_notebook_embeds_all_runtime_sources_and_parses():
    import nbformat

    notebook = build_kaggle.render()
    nbformat.validate(nbformat.from_dict(notebook))
    payload = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SOURCE_B64" for t in statement.targets):
                payload = base64.b64decode(ast.literal_eval(statement.value))
    assert payload is not None
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for folder in ("lab22", "notebooks", "scripts"):
            for path in (REPO / folder).glob("*.py"):
                assert archive.read(str(path.relative_to(REPO))) == path.read_bytes()
        assert "submission/REFLECTION.md" in archive.namelist()
        assert ".env" not in archive.namelist()


def test_result_bundle_preserves_evidence_but_excludes_credentials_weights_and_checkpoints(tmp_path):
    fixtures = {
        "adapters/dpo/adapter_config.json": '{"base_model_name_or_path":"models/sft-merged"}',
        "adapters/dpo/adapter_model.safetensors": "weight",
        "adapters/variants/dpo/split.json": "{}",
        "adapters/dpo-checkpoints/checkpoint-25/trainer_state.json": "{}",
        "models/sft-merged/config.json": "{}",
        "models/sft-merged/model.safetensors": "weight",
        "submission/logs/core.log": "real log",
        "submission/REFLECTION.md": "pending",
        "notebooks/03_dpo_train.ipynb": "{}",
        "data/pref/stats.json": "{}",
        "data/pref/train.parquet": "data",
        "data/eval/judge_summary.json": "{}",
        "gguf/model.gguf": "weight",
        ".env": "HF_TOKEN=secret",
        "credentials.token": "secret",
        ".venv/secret.txt": "secret",
    }
    for name, body in fixtures.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    target = tmp_path / "out.zip"
    runner.export_bundle(tmp_path, target)
    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
        assert {"models/sft-merged/config.json", "data/pref/stats.json", "data/pref/train.parquet",
                "adapters/variants/dpo/split.json", "submission/logs/core.log"} <= names
        assert not any("checkpoints" in name or "secret" in name or name.endswith((".safetensors", ".gguf", ".token")) for name in names)
        assert ".env" not in names
        manifest = json.loads(archive.read("RESULTS_MANIFEST.json"))
        assert set(manifest) == names - {"RESULTS_MANIFEST.json"}


def test_changed_successful_run_refuses_silent_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(runner, "signature", lambda stage: "new-signature")
    runner.save_state({"stages": {"nb0": {"status": "complete", "signature": "old-signature"}}})
    with pytest.raises(ValueError, match="changed"):
        runner.run_stage("nb0")


def test_failed_stage_keeps_error_and_exports_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(runner, "signature", lambda stage: "sig")
    exports = []
    monkeypatch.setattr(runner, "export_bundle", lambda: exports.append(True))
    def fail(stage):
        raise RuntimeError("actual GPU failure")
    monkeypatch.setattr(runner, "execute", fail)
    assert runner.run_stage("nb0") is False
    result = runner.state()["stages"]["nb0"]
    assert result["status"] == "failed"
    assert "actual GPU failure" in result["error"]
    assert exports == [True]


def test_cross_judge_never_regenerates_and_rejects_stale_outputs(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO))
    from lab22 import config as C
    from lab22 import judge as J

    monkeypatch.setattr(C, "EVAL_DIR", tmp_path)
    monkeypatch.setenv("API_JUDGE_MODEL", "user-selected-model")
    monkeypatch.setattr(J, "has_judge_key", lambda provider: True)
    (tmp_path / "side_by_side.jsonl").write_text('{"id":"e0"}\n')
    (tmp_path / "judge_results_rm.json").write_text('{"outputs_sha256":"stale"}')
    with pytest.raises(ValueError, match="stale"):
        runner.cross_judge()


def test_stage_kernel_executes_nb0_and_saves_partial_failure(tmp_path, monkeypatch):
    import shutil
    from jupyter_client.kernelspec import KernelSpecManager

    pytest.importorskip("torch")
    if "lab22-kaggle" not in KernelSpecManager().find_kernel_specs():
        pytest.skip("Optional notebook integration test needs registered lab22-kaggle kernel")
    shutil.copytree(REPO / "lab22", tmp_path / "lab22")
    folder = tmp_path / "notebooks"
    folder.mkdir()
    source = folder / "00_dpo_loss_from_scratch.py"
    shutil.copy2(REPO / "notebooks/00_dpo_loss_from_scratch.py", source)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    runner.execute("nb0")
    output = folder / "00_dpo_loss_from_scratch.ipynb"
    nb = json.loads(output.read_text())
    assert any(c.get("outputs") for c in nb["cells"])
    assert not any(o["output_type"] == "error" for c in nb["cells"] for o in c.get("outputs", []))
    source.write_text('# %%\nprint("retained")\n# %%\nraise RuntimeError("intentional failure")\n')
    with pytest.raises(Exception, match="intentional failure"):
        runner.execute("nb0")
    nb = json.loads(output.read_text())
    assert any(o["output_type"] == "error" for c in nb["cells"] for o in c.get("outputs", []))
