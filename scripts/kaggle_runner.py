"""Kaggle execution, resumable evidence collection and optional publishing.

No GPU work happens on import. The uploaded notebook embeds this exact file.
Executed stage notebooks and failure outputs are always retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = {
    "nb0": "00_dpo_loss_from_scratch",
    "sft": "01_sft_mini",
    "data": "02_preference_data",
    "dpo": "03_dpo_train",
    "eval": "04_compare_and_eval",
    "variants": "03b_dpo_variants",
    "deploy": "05_merge_deploy_gguf",
    "bench": "06_benchmark",
    "grpo": "07_grpo_bonus",
}
SETTINGS = (
    "COMPUTE_TIER", "BASE_MODEL", "MAX_LEN", "SEED", "SFT_SLICE", "PREF_TRAIN", "PREF_EVAL",
    "DPO_BETA", "DPO_LR", "DPO_EPOCHS", "JUDGE_PROMPTS", "GEN_BATCH_SIZE", "BENCH_BATCH_SIZE",
    "GEN_MAX_NEW_TOKENS", "SFT_DATASET", "PREF_DATASET", "GRPO_DATASET", "VARIANTS",
)
STATE = ROOT / "submission" / "kaggle_run.json"
REQUIRED = {
    "nb0": [],
    "sft": ["models/sft-merged/config.json", "adapters/sft-mini/adapter_config.json"],
    "data": ["data/pref/train.parquet", "data/pref/eval.parquet", "data/pref/stats.json"],
    "dpo": ["adapters/dpo/adapter_config.json", "adapters/dpo/dpo_metrics.json", "adapters/dpo/split.json"],
    "eval": ["data/eval/side_by_side.jsonl", "data/eval/judge_summary.json", "data/eval/judge_results_rm.json"],
    "variants": ["adapters/variants/variants_summary.json", "data/eval/variant_probes.json"],
    "deploy": ["data/eval/deploy_meta.json"],
    "bench": ["data/eval/benchmark_results.json"],
    "grpo": ["adapters/grpo/grpo_metrics.json", "data/eval/grpo_predictions.json"],
    "sweep": [f"adapters/dpo-b{b}/dpo_metrics.json" for b in ("0.05", "0.10", "0.50")],
    "cross": ["data/eval/judge_results_api.json", "data/eval/judge_summary.json"],
    "hub": ["data/eval/hub_publish.json"],
}
DEPENDENCIES = {
    "sft": ["nb0"], "data": ["sft"], "dpo": ["sft", "data"], "eval": ["dpo"],
    "variants": ["sft", "data"], "deploy": ["dpo"], "bench": ["dpo"],
    "grpo": ["sft"], "sweep": ["sft", "data"], "cross": ["eval"], "hub": ["eval"],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"stages": {}}


def save_state(value: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(STATE)


def signature(stage: str) -> str:
    sources = sorted((ROOT / "lab22").glob("*.py"))
    sources += [Path(__file__).resolve(), ROOT / "requirements.txt"]
    sources += [ROOT / "notebooks" / f"{STAGES[stage]}.py"] if stage in STAGES else []
    if stage == "sweep":
        sources += [ROOT / "scripts/train_dpo.py", ROOT / "scripts/eval_judge.py"]
    payload = {
        "source": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "settings": {k: os.environ.get(k) for k in SETTINGS},
        "dependencies": {s: {k: state()["stages"].get(s, {}).get(k) for k in ("signature", "artifacts")}
                         for s in DEPENDENCIES.get(stage, [])},
    }
    environment = ROOT / "submission/environment.txt"
    payload["environment"] = sha(environment) if environment.exists() else None
    if stage == "cross":
        payload["judge"] = [os.environ.get("API_JUDGE_PROVIDER"), os.environ.get("API_JUDGE_MODEL")]
    if stage == "hub":
        payload["hub"] = os.environ.get("HF_REPO")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def evidence_files(root: Path = ROOT) -> list[Path]:
    """Only return source, small evidence and executed notebooks; never credentials/weights."""
    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            continue
        if p.suffix in {".safetensors", ".bin", ".gguf", ".token", ".zip"}:
            continue
        if rel.parts[0] == "models":
            if p.name == "config.json":
                files.append(p)
            continue
        if rel.parts[0] == "adapters":
            if "checkpoints" in str(rel) or "-ckpt" in str(rel):
                continue
            if p.suffix == ".json":
                files.append(p)
            continue
        if rel.parts[0] == "data":
            if "lm_eval" in rel.parts:
                if p.name.startswith("results") and p.suffix == ".json":
                    files.append(p)
            elif p.suffix in {".json", ".jsonl", ".parquet"}:
                files.append(p)
            continue
        if rel.parts[0] in {"notebooks", "submission", "lab22", "scripts", "docs"}:
            if p.suffix in {".py", ".md", ".ipynb", ".png", ".jpg", ".json", ".txt", ".log"}:
                files.append(p)
        elif len(rel.parts) == 1 and p.name in {
            "Makefile", "requirements.txt", "LICENSE", "README.md", "rubric.md", "BONUS-CHALLENGE.md",
            "HARDWARE-GUIDE.md", "CHANGELOG.md", "VIBE-CODING.md", "pyproject.toml",
        }:
            files.append(p)
    return sorted(files)


def export_bundle(root: Path = ROOT, destination: Path | None = None) -> Path:
    destination = destination or root.parent / "LAB22_RESULTS.zip"
    files = evidence_files(root)
    manifest = {str(p.relative_to(root)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in files}
    temp = destination.with_suffix(".tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for p in files:
            archive.write(p, str(p.relative_to(root)))
        archive.writestr("RESULTS_MANIFEST.json", json.dumps(manifest, indent=2))
    temp.replace(destination)
    print(f"RESULTS: {destination} ({destination.stat().st_size / 1e6:.1f} MB)", flush=True)
    return destination


def command(args: list[str], log: Path, env: dict | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        result = subprocess.run(args, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    print(log.read_text(errors="replace")[-6000:], flush=True)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log.relative_to(ROOT)}")


def execute(stage: str) -> None:
    """Run in a separate kernel and save partially executed notebook on any error."""
    import nbformat
    from nbclient import NotebookClient
    from build_colab import percent_cells

    stem = STAGES[stage]
    cells = []
    for i, c in enumerate(percent_cells(ROOT / "notebooks" / f"{stem}.py")):
        c["id"] = f"{stage}-{i:03d}"
        c["source"] = "".join(c["source"])
        cells.append(nbformat.from_dict(c))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "lab22-kaggle", "display_name": "Lab22 Kaggle", "language": "python"}
    output = ROOT / "notebooks" / f"{stem}.ipynb"

    def persist(*args, **kwargs):
        nbformat.write(nb, output)

    def progress(cell, cell_index, **kwargs):
        print(f"{stage}: cell {cell_index + 1}/{len(nb.cells)}", flush=True)

    client = NotebookClient(
        nb, timeout=None, kernel_name="lab22-kaggle", resources={"metadata": {"path": str(ROOT)}},
        on_cell_execute=progress, on_cell_executed=persist,
    )
    try:
        client.execute()
    finally:
        persist()


def cross_judge() -> None:
    """Reuse saved NB4 outputs exactly; never regenerate answers for cross judging."""
    sys.path.insert(0, str(ROOT))
    from lab22 import config as C
    from lab22 import judge as J

    provider, model = os.environ.get("API_JUDGE_PROVIDER", "gemini"), os.environ.get("API_JUDGE_MODEL", "")
    if not model or not J.has_judge_key(provider):
        raise ValueError("API_JUDGE_MODEL or matching Kaggle Secret is missing")
    outputs = C.EVAL_DIR / "side_by_side.jsonl"
    digest = sha(outputs)
    rm = json.loads((C.EVAL_DIR / "judge_results_rm.json").read_text())
    if rm["outputs_sha256"] != digest:
        raise ValueError("RM results are stale; rerun NB4 before cross judging")
    records = [json.loads(line) for line in outputs.read_text().splitlines() if line.strip()]
    caller = J.make_caller(provider, model)
    judged = []
    for i, r in enumerate(records, 1):
        judged.append({**r, **J.judge_pair(r["prompt"], r["sft"], r["dpo"], caller)})
        print(f"API judge {i}/{len(records)}", flush=True)
        (C.EVAL_DIR / "judge_results_api_partial.json").write_text(
            json.dumps({"outputs_sha256": digest, "records": judged}, ensure_ascii=False, indent=2)
        )
    api = {"judge": f"{provider}:{model}", "outputs_sha256": digest, "records": judged}
    (C.EVAL_DIR / "judge_results_api.json").write_text(json.dumps(api, ensure_ascii=False, indent=2))
    summary = json.loads((C.EVAL_DIR / "judge_summary.json").read_text())
    if summary["outputs_sha256"] != digest:
        raise ValueError("NB4 summary no longer matches the saved answers")
    summary["cross_judge"] = {"other_judge": api["judge"], **J.agreement(rm["records"], judged)}
    summary["api_judge"] = {
        "judge": api["judge"],
        **{c: J.summarize([r for r in judged if r["category"] == c], seed=C.SEED)
           for c in ("heldout", "helpfulness", "safety")},
    }
    (C.EVAL_DIR / "judge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary["cross_judge"], indent=2), flush=True)
    if summary["api_judge"]["heldout"]["n"] < 50:
        raise ValueError("Fewer than 50 valid API held-out verdicts; inspect API failures")


def publish_hub() -> None:
    """Publish the actual SFT reference, then a separately rewritten DPO adapter."""
    import shutil
    from huggingface_hub import HfApi

    repo = os.environ.get("HF_REPO", "")
    token = os.environ.get("HF_TOKEN")
    if not token or repo.count("/") != 1:
        raise ValueError("HF_TOKEN and HF_REPO=username/model-name are required")
    if os.environ.get("PUBLISH_HF", "0") != "1":
        raise ValueError("Set PUBLISH_HF=True in notebook settings to publish public models")
    reference = repo + "-sft-reference"
    api = HfApi(token=token)
    metrics = json.loads((ROOT / "adapters/dpo/dpo_metrics.json").read_text())
    summary = json.loads((ROOT / "data/eval/judge_summary.json").read_text())
    source_config = json.loads((ROOT / "adapters/dpo/adapter_config.json").read_text())
    model_card = (
        "---\nlicense: other\nlicense_name: upstream-terms-and-noncommercial-training-data\n"
        "license_link: https://huggingface.co/datasets/saillab/alpaca-vietnamese-cleaned\n"
        "language:\n- vi\ntags:\n- peft\n- dpo\n- educational\n"
        f"base_model: {reference}\n---\n\n# Lab22 Vietnamese SFT+DPO (experimental)\n\n"
        "Educational experiment. Not evaluated for production. SFT data is CC BY-NC according to the lab; "
        "use for non-commercial education/research and review upstream terms.\n\n"
        f"Original base: `{metrics['base_model']}`. Actual DPO reference: `{reference}`.\n\n"
        "SFT: saillab/alpaca-vietnamese-cleaned. Preferences: sailor2/sea-ultrafeedback-onpolicy, Vietnamese. "
        "Chosen/rejected labels are model-generated; length bias and judge-family bias are possible.\n\n"
        f"## Training\n\n```json\n{json.dumps(metrics, indent=2)}\n```\n\n"
        f"LoRA configuration: `{json.dumps({k: source_config.get(k) for k in ('r', 'lora_alpha', 'target_modules')})}`.\n\n"
        f"## Held-out evaluation\n\n```json\n{json.dumps(summary.get('heldout'), indent=2)}\n```\n\n"
        "Report confidence intervals, sanity accuracy and length bias; an observed win rate does not guarantee quality.\n\n"
        "## Load\n\n```python\nfrom transformers import AutoTokenizer, AutoModelForCausalLM\n"
        "from peft import PeftModel\n"
        f"tokenizer = AutoTokenizer.from_pretrained('{repo}')\n"
        f"base = AutoModelForCausalLM.from_pretrained('{reference}', device_map='auto', dtype='auto')\n"
        f"model = PeftModel.from_pretrained(base, '{repo}')\n```\n\n"
        "Use the tokenizer chat template with enable_thinking=False. Limitations: small training/evaluation sets, "
        "potential hallucinations, no production safety certification, no commercial-use claim.\n"
    )
    # Local grading config remains untouched. Public distribution gets its own directory.
    export = ROOT.parent / "lab22-hub-export"
    export.mkdir(exist_ok=True)
    for p in (ROOT / "adapters/dpo").iterdir():
        if p.is_file() and p.suffix in {".json", ".safetensors", ".model", ".txt", ".jinja"}:
            shutil.copy2(p, export / p.name)
    source_config["base_model_name_or_path"] = reference
    (export / "adapter_config.json").write_text(json.dumps(source_config, indent=2))
    (export / "README.md").write_text(model_card)
    ref_card = ROOT / "models/sft-merged/README.md"
    ref_card.write_text(
        "---\nlicense: other\nlicense_name: upstream-terms-and-noncommercial-training-data\n"
        "license_link: https://huggingface.co/datasets/saillab/alpaca-vietnamese-cleaned\n"
        "language:\n- vi\n---\n\n# Lab22 SFT reference\n\n"
        f"Merged reference for [{repo}](https://huggingface.co/{repo}).\n\n"
        f"Original base: {metrics['base_model']}. SFT data: saillab/alpaca-vietnamese-cleaned "
        "(CC BY-NC per lab documentation), 1 epoch. Experimental, non-commercial educational use. "
        "Known limitations: translated data, small sample, hallucinations.\n"
    )
    for name, folder in ((reference, ROOT / "models/sft-merged"), (repo, export)):
        api.create_repo(name, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(repo_id=name, folder_path=str(folder), commit_message="Publish Lab22 educational experiment")
        published = api.list_repo_files(name)
        if not any(p.endswith(".safetensors") for p in published):
            raise ValueError(f"Published repo {name} has no model weights")
    (ROOT / "data/eval/hub_publish.json").write_text(
        json.dumps({"adapter_url": f"https://huggingface.co/{repo}",
                    "reference_url": f"https://huggingface.co/{reference}"}, indent=2)
    )
    print(f"Published https://huggingface.co/{repo}", flush=True)


def artifacts_exist(stage: str) -> bool:
    if not all((ROOT / p).is_file() for p in REQUIRED[stage]):
        return False
    if stage in STAGES and not (ROOT / "notebooks" / f"{STAGES[stage]}.ipynb").is_file():
        return False
    weight_dirs = {"sft": "models/sft-merged", "dpo": "adapters/dpo"}
    if stage in weight_dirs and not list((ROOT / weight_dirs[stage]).glob("*.safetensors")):
        return False
    return True


def artifact_hashes(stage: str) -> dict[str, str]:
    paths = list(REQUIRED[stage])
    if stage == "eval":
        # Cross judging enriches this summary later; saved RM verdicts and outputs stay immutable.
        paths.remove("data/eval/judge_summary.json")
    if stage in STAGES:
        paths.append(f"notebooks/{STAGES[stage]}.ipynb")
    return {p: sha(ROOT / p) for p in paths if (ROOT / p).exists()}


def run_stage(stage: str) -> bool:
    value = state()
    marker = value["stages"].get(stage, {})
    digest = signature(stage)
    if (marker.get("status") == "complete" and marker.get("signature") == digest and artifacts_exist(stage)
            and marker.get("artifacts") == artifact_hashes(stage)):
        print(f"SKIP {stage}: matching successful run and artifacts", flush=True)
        return True
    if marker.get("status") == "complete" and marker.get("signature") != digest:
        raise ValueError(f"{stage}: sources/settings/dependencies changed. Use a fresh experiment directory; do not silently overwrite.")
    missing = [s for s in DEPENDENCIES.get(stage, []) if value["stages"].get(s, {}).get("status") != "complete"]
    if missing:
        value["stages"][stage] = {"status": "blocked", "reason": f"Dependencies incomplete: {missing}"}
        save_state(value)
        print(f"BLOCKED {stage}: {missing}", flush=True)
        export_bundle()
        return False
    started = time.time()
    value["stages"][stage] = {"status": "running", "signature": digest}
    save_state(value)
    print(f"START {stage}", flush=True)
    try:
        if stage in STAGES:
            if stage == "deploy":
                env = dict(os.environ, CMAKE_ARGS="-DGGML_CUDA=on")
                command([sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.3.16,<1.0"],
                        ROOT / "submission/logs/llama-install.log", env)
            execute(stage)
        elif stage == "sweep":
            env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
            command(["make", "beta-sweep"], ROOT / "submission/logs/beta-sweep.log", env)
        elif stage == "cross":
            cross_judge()
        elif stage == "hub":
            publish_hub()
        if not artifacts_exist(stage):
            raise ValueError(f"{stage}: expected output artifacts are missing after execution")
        value = state()
        value["stages"][stage] = {"status": "complete", "signature": digest, "seconds": time.time() - started,
                                  "artifacts": artifact_hashes(stage)}
        save_state(value)
        return True
    except Exception as exc:
        value = state()
        value["stages"][stage] = {"status": "failed", "signature": digest,
                                  "error": f"{type(exc).__name__}: {exc}"[-5000:], "seconds": time.time() - started}
        save_state(value)
        print(f"FAILED {stage}: {exc}", flush=True)
        return False
    finally:
        export_bundle()


def audit() -> None:
    """Keep the real verifier output; reflection remains unfinished until results are reviewed."""
    sys.path.insert(0, str(ROOT))
    value = state()
    outputs = ROOT / "data/eval/side_by_side.jsonl"
    report = {"stages": value["stages"], "reflection_pending": True, "gguf_manual_screenshot_pending": True}
    if outputs.exists():
        rows = [json.loads(line) for line in outputs.read_text().splitlines() if line.strip()]
        held = [r for r in rows if r["category"] == "heldout"]
        report["distinct_heldout_outputs"] = len({" ".join(r["prompt"].lower().split()) for r in held})
    summary = ROOT / "data/eval/judge_summary.json"
    if summary.exists():
        report["judge_heldout"] = json.loads(summary.read_text()).get("heldout")
    report["notes"] = ["No personal reflection or performance claims have been fabricated.",
                       "A completed run is not a guarantee of maximum marks."]
    (ROOT / "submission/collection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    result = subprocess.run([sys.executable, str(ROOT / "scripts/verify.py")], capture_output=True, text=True, cwd=ROOT)
    (ROOT / "submission/verify-before-reflection.log").write_text(result.stdout + result.stderr)
    print(result.stdout, flush=True)
    print("Verifier exit:", result.returncode, "(reflection is intentionally pending)", flush=True)
    export_bundle()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=[*STAGES, "sweep", "cross", "hub", "audit", "export"])
    args = parser.parse_args()
    if args.stage == "audit":
        audit()
        return 0
    if args.stage == "export":
        export_bundle()
        return 0
    return 0 if run_stage(args.stage) else 1


if __name__ == "__main__":
    raise SystemExit(main())
