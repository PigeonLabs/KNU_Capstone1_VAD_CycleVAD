"""Small, atomic, inspectable run artifacts."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()

def jsonable(v):
    if isinstance(v, dict): return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray): return jsonable(v.tolist())
    if isinstance(v, np.generic): return jsonable(v.item())
    if isinstance(v, float) and not np.isfinite(v): return None
    if isinstance(v, Path): return str(v)
    return v

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    tmp.replace(path)

def read_json(path): return json.loads(Path(path).read_text())

def environment():
    import torch
    packages = {}
    for name in ["torch", "torchvision", "transformers", "numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "Pillow", "opencv-python-headless", "nbformat", "nbclient"]:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

def source_identity():
    files = sorted(p for folder in ["src", "configs", "vendor"] for p in (ROOT/folder).rglob("*") if p.is_file() and p.suffix in [".py", ".json"])
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in files}
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": digest(hashes), "source_files": hashes}

def stamp(): return datetime.now(timezone.utc).isoformat()

def log(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    record = jsonable({"time_utc": stamp(), **values})
    with path.open("a") as f: f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)

def init_run(name, cfg):
    out = ROOT/"results"/name; out.mkdir(parents=True, exist_ok=True)
    if (out/"config.json").exists() and digest(read_json(out/"config.json")) != digest(cfg):
        raise ValueError(f"Run config changed: {name}; use a new run id")
    write_json(out/"config.json", cfg)
    write_json(out/"environment.json", environment())
    path = out/"run_manifest.json"
    if not path.exists():
        write_json(path, {"run_id": name, "started_utc": stamp(), "status": "running", **source_identity()})
    return out

def complete_run(out, **values):
    path = Path(out)/"run_manifest.json"
    write_json(path, {**read_json(path), "status": "completed", "completed_utc": stamp(), **values})
