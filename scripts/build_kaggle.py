"""Generate one uploadable Kaggle notebook with a frozen, embedded source bundle."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def render(config: dict | None = None) -> dict:
    paths = []
    for folder in ("lab22", "notebooks", "scripts", "docs", "submission", "colab"):
        paths.extend(p for p in (REPO / folder).rglob("*") if p.is_file()
                     and p.suffix in ({".py"} if folder == "notebooks" else {".py", ".md", ".ipynb"}))
    paths.extend(REPO / name for name in (
        "Makefile", "requirements.txt", "pyproject.toml", ".gitignore", "LICENSE", "README.md", "rubric.md",
        "HARDWARE-GUIDE.md", "BONUS-CHALLENGE.md", "CHANGELOG.md", "VIBE-CODING.md",
    ))
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(paths):
            if "__pycache__" not in p.parts:
                # Stable timestamp makes repeated builds reproducible.
                info = zipfile.ZipInfo(str(p.relative_to(REPO)), (2026, 10, 8, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, (REPO / "submission/REFLECTION.template.md").read_bytes()
                                 if str(p.relative_to(REPO)) == "submission/REFLECTION.md"
                                 and (REPO / "submission/REFLECTION.template.md").exists() else p.read_bytes())
    raw = payload.getvalue()
    encoded = base64.b64encode(raw).decode()
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True),
                      "execution_count": None, "outputs": []})

    md("""# Lab 22 — Kaggle · phần chính + toàn bộ bonus có điểm

**Tác giả bài nộp: Trần Nguyễn Tiến Đức · 2A202602871 · K4.** Sửa thông tin nếu cần.

1. Upload file này vào Kaggle Notebook; bật **Internet**, chọn **GPU T4 ×2**.
2. Chạy từ trên xuống. Mỗi stage dùng kernel riêng; notebook có output được giữ trong `/kaggle/working/lab22/notebooks/`.
3. Mỗi stage xuất `/kaggle/working/LAB22_RESULTS.zip`. **Tải ZIP này ngay sau phần chính và khi kết thúc.**
4. Gửi ZIP + notebook Kaggle đã chạy + ảnh màn hình NB5 cho Codex. REFLECTION được hoàn thiện sau khi đọc số liệu thật.

Notebook chứa nguồn cố định, không cần clone GitHub. Mặc định chạy NB0–NB4, 5 biến thể, GGUF, benchmark, GRPO và β-sweep.
Chấm chéo cần API secret + model ID; HF Hub cần HF secret + repo ID và bật PUBLISH_HF. Thiếu cấu hình sẽ ghi **skipped**, không tính là đã làm.
Đây là gói chuẩn bị; GPU end-to-end chưa được kiểm chứng trên Kaggle. Nếu gặp lỗi, các output dở dang vẫn nằm trong ZIP để chẩn đoán.
""")
    md("## A. Cấu hình — sửa trước lần chạy đầu tiên\n\nKhông đổi MAX_LEN/seed/dữ liệu giữa các stage. T4 ×2 vẫn dùng tier T4, chương trình chỉ thấy một GPU.")
    code('''import os
from pathlib import Path

ROOT = Path("/kaggle/working/lab22")
VENV = Path("/tmp/lab22-kaggle-venv")
NOTEBOOK_VERSION = "kaggle-v2-pip-bootstrap"
RESTORE_ARCHIVE = ""  # Optional: exact /kaggle/input/.../LAB22_CHECKPOINT.zip path
RUN_BONUS = True
API_JUDGE_PROVIDER = "gemini"  # gemini | openai | anthropic
API_JUDGE_MODEL = ""  # Enter a current model ID available to your API key
HF_REPO = ""  # e.g. your-hf-username/lab22-dpo-vn-experimental
PUBLISH_HF = False  # True authorizes upload of TWO public model repos, adapter + SFT reference
CREATE_CHECKPOINT_ZIP = False  # Extra archive with weights; requires substantial free disk space

SETTINGS = {
    "CUDA_VISIBLE_DEVICES": "0", "COMPUTE_TIER": "T4", "MAX_LEN": "768", "SEED": "42",
    "SFT_SLICE": "1000", "PREF_TRAIN": "800", "PREF_EVAL": "100", "JUDGE_PROMPTS": "80",
    "DPO_BETA": "0.1", "DPO_LR": "5e-6", "DPO_EPOCHS": "1",
    "GEN_BATCH_SIZE": "2", "BENCH_BATCH_SIZE": "1", "JUDGE_PROVIDER": "rm",
    "API_JUDGE_PROVIDER": API_JUDGE_PROVIDER, "API_JUDGE_MODEL": API_JUDGE_MODEL,
    "HF_REPO": HF_REPO, "PUBLISH_HF": "1" if PUBLISH_HF else "0",
    "TOKENIZERS_PARALLELISM": "false", "HF_HOME": "/tmp/lab22-hf-cache",
}
os.environ.update(SETTINGS)
print("Notebook version:", NOTEBOOK_VERSION)
print("Run root:", ROOT)
print("API cross judge configured:", bool(API_JUDGE_MODEL))
print("Public HF publishing enabled:", PUBLISH_HF)
''')
    md("## B. Khôi phục checkpoint nếu chuyển phiên, rồi giải nén nguồn\n\nĐể trống RESTORE_ARCHIVE ở lần chạy đầu. Chỉ khôi phục archive checkpoint do notebook này tạo; ZIP kết quả nhỏ không chứa trọng số.")
    code('''import base64, hashlib, io, json, zipfile

ROOT.mkdir(parents=True, exist_ok=True)

def safe_extract(archive, root, preserve_source=False):
    for item in archive.infolist():
        destination = (root / item.filename).resolve()
        if not destination.is_relative_to(root.resolve()):
            raise ValueError("Unsafe path in archive")
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        body = archive.read(item)
        if preserve_source and destination.exists() and destination.read_bytes() != body:
            raise ValueError(f"Existing source differs: {item.filename}. Preserve your experiment; use a fresh root.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or not preserve_source:
            destination.write_bytes(body)

if RESTORE_ARCHIVE:
    if (ROOT / "submission/kaggle_run.json").exists():
        print("Existing experiment found; keeping it instead of overwriting with checkpoint")
    else:
        with zipfile.ZipFile(RESTORE_ARCHIVE) as checkpoint:
            safe_extract(checkpoint, ROOT)
'''+f'''
SOURCE_B64 = {encoded!r}
payload = base64.b64decode(SOURCE_B64)
assert hashlib.sha256(payload).hexdigest() == {hashlib.sha256(raw).hexdigest()!r}
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    safe_extract(archive, ROOT, preserve_source=True)
os.chdir(ROOT)
print("Source bundle ready. No GitHub clone required.")
import importlib.util
collector_spec = importlib.util.spec_from_file_location("lab22_collector", ROOT / "scripts/kaggle_runner.py")
collector = importlib.util.module_from_spec(collector_spec)
collector_spec.loader.exec_module(collector)
(ROOT / "submission/run_settings.json").write_text(json.dumps(SETTINGS, indent=2))
collector.export_bundle()
''')
    md("## C. Secrets (chỉ cần cho chấm chéo và HF Hub)\n\nTrong Kaggle Secrets, thêm key đúng tên GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY và HF_TOKEN khi cần; bật quyền cho notebook. Không in secret.")
    code('''try:
    from kaggle_secrets import UserSecretsClient
    client = UserSecretsClient()
    names = []
    if API_JUDGE_MODEL:
        names.append({"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[API_JUDGE_PROVIDER])
    if PUBLISH_HF:
        names.append("HF_TOKEN")
    for name in names:
        try:
            os.environ[name] = client.get_secret(name)
            print(name, "loaded")
        except Exception:
            print(name, "MISSING — corresponding bonus cannot run")
except ImportError:
    print("Kaggle Secrets unavailable; base lab needs no API key")
''')
    md("## D. Cài môi trường riêng\n\nMôi trường Python riêng giúp tránh sửa các package mà kernel Kaggle đang dùng. Lần đầu có thể lâu vì tải torch/Unsloth. Không tắt Internet giữa chừng.")
    code('''import subprocess, sys

print("Setup:", NOTEBOOK_VERSION)
VENV.parent.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
from kaggle_setup import ensure_pip
for name in ("PYTHONPATH", "PYTHONHOME"):
    os.environ.pop(name, None)
PY = ensure_pip(VENV)
requirements = (ROOT / "requirements.txt").read_text()
core = "\\n".join(line for line in requirements.splitlines()
                 if not line.strip().startswith(("llama-cpp-python", "jupyterlab")))
(ROOT / "kaggle-runtime-requirements.txt").write_text(core + "\\nwrapt>=1.16,<3\\nnbclient>=0.10,<1\\nnbformat>=5.10,<6\\nipykernel>=6,<8\\n")
install_log = ROOT / "submission/logs/install.log"
install_log.parent.mkdir(parents=True, exist_ok=True)
with install_log.open("w") as log:
    result = subprocess.run([PY, "-m", "pip", "install", "-r", str(ROOT / "kaggle-runtime-requirements.txt")],
                            stdout=log, stderr=subprocess.STDOUT)
print(install_log.read_text(errors="replace")[-4000:])
collector.export_bundle()
if result.returncode:
    raise RuntimeError("Install failed. Download install.log and send it with LAB22_RESULTS.zip.")
subprocess.run([PY, "-m", "ipykernel", "install", "--prefix", str(VENV),
                "--name", "lab22-kaggle", "--display-name", "Lab22 Kaggle"], check=True)
os.environ["JUPYTER_PATH"] = str(VENV / "share/jupyter") + os.pathsep + os.environ.get("JUPYTER_PATH", "")
with (ROOT / "submission/environment.txt").open("w") as log:
    subprocess.run([PY, "-m", "pip", "freeze"], stdout=log, check=True)
subprocess.run(["nvidia-smi"], check=True)
subprocess.run([PY, "scripts/verify.py", "--smoke"], check=True, cwd=ROOT)
print("Environment ready")
''')
    md("## E. Bộ điều khiển và gói kết quả\n\nCác stage đã hoàn thành sẽ được bỏ qua chỉ khi nguồn/cấu hình và artifact còn khớp. Bonus lỗi không xóa phần chính. REFLECTION giữ trạng thái chờ phân tích.")
    code('''import importlib.util, shutil

spec = importlib.util.spec_from_file_location("lab22_collector", ROOT / "scripts/kaggle_runner.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)

def run(stage, required=False):
    log = ROOT / "submission/logs" / f"{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        proc = subprocess.Popen([PY, "-u", "scripts/kaggle_runner.py", stage], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            result = proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            collector.export_bundle()
    if result and required:
        raise RuntimeError(f"Core stage {stage} failed. Download LAB22_RESULTS.zip; do not continue dependent training.")
    return result == 0

def record_skip(stage, reason):
    value = collector.state()
    if value["stages"].get(stage, {}).get("status") != "complete":
        value["stages"][stage] = {"status": "skipped", "reason": reason}
        collector.save_state(value)
    print("SKIPPED", stage, reason)
    collector.export_bundle()

collector.export_bundle()
''')
    for stage, label in (("nb0", "NB0 — loss tự triển khai và assert"), ("sft", "NB1 — SFT và reference merged"),
                         ("data", "NB2 — split, 5 cặp mẫu, length bias"), ("dpo", "NB3 — DPO và reward train/held-out"),
                         ("eval", "NB4 — 8 câu cố định + 80 held-out dự kiến, RM panel")):
        md("## " + label + "\n\nXem output đầy đủ trong file notebook tương ứng ở thư mục `notebooks/` của Output.")
        code(f"run({stage!r}, required=True)")
    md("## Lưu phần chính ngay\n\nTải LAB22_RESULTS.zip trước khi tiếp tục. Nếu chuyển phiên, cần cả weights SFT/DPO, không chỉ ZIP nhỏ. Giữ thư mục lab22 qua Saved Output hoặc checkpoint tùy chọn cuối notebook.")
    code("collector.export_bundle()\nprint('CORE FINISHED — download /kaggle/working/LAB22_RESULTS.zip now')")
    for stage, label in (("variants", "NB3b — năm biến thể (+8)"), ("deploy", "NB5 — GGUF Q4_K_M (+4)"),
                         ("bench", "NB6 — ba benchmark (+6)"), ("grpo", "NB7 — GRPO (+8)"),
                         ("sweep", "β-sweep — ba beta (+6)")):
        md("## " + label + ("\n\n**Chụp ảnh màn hình kết quả llama.cpp trong notebook NB5 đã chạy.** Ảnh thủ công còn cần khi nộp." if stage == "deploy" else ""))
        code(f"if RUN_BONUS:\n    run({stage!r})\nelse:\n    record_skip({stage!r}, 'RUN_BONUS=False')")
    md("## Chấm chéo (+4) — dùng đúng đầu ra NB4 đã lưu\n\nKhông sinh lại câu trả lời. Cần API model ID và Secret. Chi phí/rate limit tùy tài khoản provider.")
    code('''if RUN_BONUS and API_JUDGE_MODEL:
    run("cross")
else:
    record_skip("cross", "API_JUDGE_MODEL missing or bonuses disabled")
''')
    md("## HF Hub (+3) — xuất bản adapter và SFT reference\n\nBật PUBLISH_HF ở đầu notebook chỉ khi muốn upload hai repo public, có HF_REPO và HF_TOKEN. Bản adapter local dùng chấm bài không bị đổi đường reference.")
    code('''if RUN_BONUS and PUBLISH_HF and HF_REPO:
    run("hub")
else:
    record_skip("hub", "PUBLISH_HF disabled or HF_REPO missing")
''')
    md("## F. Kiểm kê và tải kết quả\n\nVerify trước phản tư có thể báo UNEDITED — đây là đúng trạng thái. Không ghi kết quả giả để làm verify xanh. Gửi ZIP cho Codex để hoàn thiện phản tư bằng số thật và phân tích các bonus lỗi/thiếu.")
    code('''run("audit")
bundle = collector.export_bundle()
from IPython.display import FileLink, display
display(FileLink(str(bundle)))
print("Download from Output:", bundle)
print("Also save/download this Kaggle notebook with outputs.")
''')
    md("## G. Checkpoint có weights (tùy chọn)\n\nZIP này lớn, khác ZIP nộp bài. Chỉ bật CREATE_CHECKPOINT_ZIP nếu đủ dung lượng để tạo thêm một bản sao. Khi chuyển phiên, upload checkpoint thành Dataset private, đặt RESTORE_ARCHIVE đến file đó; dùng cùng bản notebook này.")
    code('''if CREATE_CHECKPOINT_ZIP:
    files = [p for p in ROOT.rglob("*") if p.is_file()
             and not any(part.startswith(".") or "checkpoints" in part or part.endswith("-ckpt")
                         or part == "__pycache__" for part in p.relative_to(ROOT).parts)]
    size = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(ROOT.parent).free < size + 2 * 1024**3:
        raise RuntimeError("Not enough free disk for checkpoint copy. Keep original weights through saved output instead.")
    destination = ROOT.parent / "LAB22_CHECKPOINT.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_STORED) as archive:
        for p in files:
            archive.write(p, str(p.relative_to(ROOT)))
    print("Checkpoint:", destination, f"{destination.stat().st_size / 1e9:.2f} GB")
else:
    print("No duplicate checkpoint ZIP created. Preserve lab22/models and lab22/adapters in saved outputs.")
''')
    for i, cell in enumerate(cells):
        cell["id"] = f"lab22-{i:03d}"
    if config:
        settings = cells[2]
        source = "".join(settings["source"])
        for key, value in config.items():
            if key not in {"ROOT", "VENV", "API_JUDGE_PROVIDER", "API_JUDGE_MODEL", "HF_REPO", "PUBLISH_HF"}:
                raise ValueError(f"Unsupported notebook setting: {key}")
            literal = f"Path({value!r})" if key in {"ROOT", "VENV"} else repr(value)
            source, count = re.subn(rf"^{key} = .*?$", lambda _: f"{key} = {literal}", source, count=1, flags=re.MULTILINE)
            assert count == 1, key
        settings["source"] = source.splitlines(keepends=True)
        if "ROOT" in config:
            for cell in cells:
                if cell["cell_type"] == "markdown":
                    cell["source"] = "".join(cell["source"]).replace("/kaggle/working/lab22", config["ROOT"]).splitlines(keepends=True)
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    folder = REPO / "kaggle"
    folder.mkdir(exist_ok=True)
    target = folder / "Lab22_Kaggle_All.ipynb"
    target.write_text(json.dumps(render(), ensure_ascii=False, indent=1) + "\n")
    print(f"Generated {target} ({target.stat().st_size / 1e3:.0f} KB)")
    fixed = folder / "Lab22_Kaggle_All_Fixed.ipynb"
    fixed.write_text(json.dumps(render({
        "ROOT": "/kaggle/working/lab22-v2", "VENV": "/tmp/lab22-kaggle-venv-v2",
        "API_JUDGE_PROVIDER": "openai", "API_JUDGE_MODEL": "gpt-4.1-mini",
        "HF_REPO": "duckak12/lab22-dpo-vn-experimental", "PUBLISH_HF": True,
    }), ensure_ascii=False, indent=1) + "\n")
    print(f"Generated configured repair notebook: {fixed}")


if __name__ == "__main__":
    main()
