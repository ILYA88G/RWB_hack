"""Reproduce the five-fold study, reusing completed component checkpoints."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAGES = [
    ("components.py", "cat"),
    ("components.py", "upstream"),
    ("temporal.py",),
    ("experiment.py", "build", "2025-10-11", "2025-10-13"),
    ("experiment.py", "select"),
    ("experiment.py", "build", "2025-10-18", "2025-10-20", "2025-10-25"),
    ("experiment.py", "assess"),
    ("finalize.py",),
]

if __name__ == "__main__":
    for script, *args in STAGES:
        subprocess.run([sys.executable, "-B", "-u", str(HERE/script), *args], cwd=ROOT, check=True)
