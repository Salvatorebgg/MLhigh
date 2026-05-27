from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "app" / "static"
DATA_DIR = ROOT / "data"
EXAMPLES_DIR = DATA_DIR / "examples"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = ROOT / "outputs"

for d in [EXAMPLES_DIR, UPLOADS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = 200
