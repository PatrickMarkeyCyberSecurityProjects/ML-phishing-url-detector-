"""Assemble the Hugging Face Space bundle in build/space/.

A Docker Space needs its Dockerfile and README.md at the root of the Space
repository, and the app needs `src/features.py`, the model files and
`reports/metrics.json` beside it. This script copies exactly those files, so
the folder it writes is both the Docker build context and the content of the
Space repository.

    python app/build_space.py                      # writes build/space/
    docker build -t phishing-url-scorer build/space
    python app/build_space.py path/to/space-clone  # fills a clone of the Space repository

In a clone, `.git` and `.gitattributes` are kept; every other file there is
replaced by the bundle.

Only the standard library is used. Nothing is downloaded or uploaded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "build" / "space"

#: source path in the repository -> path in the bundle
FILES = {
    "app/README.md": "README.md",
    "app/Dockerfile": "Dockerfile",
    "app/requirements.txt": "requirements.txt",
    "app/app.py": "app.py",
    "app/background.js": "background.js",
    "src/__init__.py": "src/__init__.py",
    "src/features.py": "src/features.py",
    "models/lgbm.txt": "models/lgbm.txt",
    "models/cnn.onnx": "models/cnn.onnx",
    "models/lgbm_operating.json": "models/lgbm_operating.json",
    "models/cnn_operating.json": "models/cnn_operating.json",
    "models/review_operating.json": "models/review_operating.json",
    "models/hashes.json": "models/hashes.json",
    "reports/metrics.json": "reports/metrics.json",
}
#: files whose hashes were recorded when the models were frozen or the review point was added
HASHED = ("models/lgbm.txt", "models/cnn.onnx", "models/lgbm_operating.json", "models/cnn_operating.json",
          "models/review_operating.json", "src/features.py")


#: left alone when the target folder is a clone of the Space repository
KEEP = (".git", ".gitattributes")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out: Path = DEFAULT_OUT) -> Path:
    recorded = json.loads((ROOT / "models" / "hashes.json").read_text(encoding="utf-8"))
    for name in HASHED:
        if sha256(ROOT / name) != recorded[name]["sha256"]:
            raise SystemExit(f"{name} does not match models/hashes.json: refusing to bundle a changed file")
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    if "post_hoc" not in metrics:
        raise SystemExit("reports/metrics.json has no post_hoc key: run notebooks/05_post_hoc.ipynb first")

    if out.exists():
        looks_right = (out / "Dockerfile").exists() or (out / ".git").exists() or not any(out.iterdir())
        if out.resolve() != DEFAULT_OUT.resolve() and not looks_right:
            raise SystemExit(f"{out} exists and is neither a Space bundle nor a git clone: not touching it")
        for child in out.iterdir():          # in a clone of the Space, keep git's own files
            if child.name in KEEP:
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    for source, target in FILES.items():
        destination = out / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, destination)
    return out


if __name__ == "__main__":
    target = build(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUT)
    files = sorted(p for p in target.rglob("*") if p.is_file() and not set(p.relative_to(target).parts) & set(KEEP))
    total = sum(p.stat().st_size for p in files)
    for p in files:
        print(f"{p.stat().st_size:>10,}  {p.relative_to(target).as_posix()}")
    print(f"{total:>10,}  bytes in {len(files)} files -> {target}")
