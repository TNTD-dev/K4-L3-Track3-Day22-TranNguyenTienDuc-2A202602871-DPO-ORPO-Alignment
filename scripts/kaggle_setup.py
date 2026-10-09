"""Bootstrap pip even in a pre-existing, partially initialized Kaggle venv."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def clean_environment() -> dict[str, str]:
    # Do not carry the host image's Python import hooks into the isolated interpreter.
    return {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}


def ensure_pip(venv: Path, host_python: str = sys.executable) -> str:
    env = clean_environment()
    python = str(venv / "bin/python")
    if not Path(python).exists():
        subprocess.run([host_python, "-I", "-m", "venv", "--without-pip", str(venv)], env=env, check=True)
    probe = subprocess.run([python, "-I", "-m", "pip", "--version"], env=env, capture_output=True, text=True)
    if probe.returncode:
        seeded = subprocess.run([python, "-I", "-m", "ensurepip", "--upgrade"],
                                env=env, capture_output=True, text=True)
        if seeded.returncode:
            # pip --python is explicitly supported for target environments without pip.
            subprocess.run([host_python, "-I", "-m", "pip", "--python", python, "install", "pip"],
                           env=env, check=True)
    subprocess.run([python, "-I", "-m", "pip", "--version"], env=env, check=True)
    return python
