"""Frame-exact manifest; no guessed label repair or random frame splitting."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .io import ROOT, write_json, read_json, file_hash, digest, init_run, complete_run

PARTS = ["fit", "validation", "reference", "threshold"]

def frame_paths(root, row):
    folder = Path(root)/row["scene"]/row["source_split"]/"frames"/row["video"]
    paths = sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem))
    if [int(p.stem) for p in paths] != list(range(row["frames"])):
        raise ValueError(f"Missing/reordered frames: {row['id']}")
    return paths

def allocations(n, fractions):
    target = n*np.asarray(fractions); counts = np.floor(target).astype(int)
    for i in np.argsort(-(target-counts), kind="stable")[:n-counts.sum()]: counts[i] += 1
    return counts.tolist()

def partition(rows, cfg):
    result = []
    for scene in cfg["scenes"]:
        rng = np.random.default_rng(cfg["split_seed"])
        training = sorted([r for r in rows if r["scene"] == scene and r["source_split"] == "training"], key=lambda r: int(r["video"]))
        order = rng.permutation(len(training)); offset = 0
        for part, count in zip(PARTS, allocations(len(training), cfg["normal_fractions"])):
            for i in order[offset:offset+count]: result.append({**training[i], "partition": part})
            offset += count
        for stratum in ["normal", "mixed", "anomaly"]:
            test = sorted([r for r in rows if r["scene"] == scene and r["source_split"] == "testing" and r["stratum"] == stratum and r["valid"]], key=lambda r: int(r["video"]))
            order = rng.permutation(len(test))
            ndev = int(np.floor(len(test)*cfg["development_fraction"]+.5))
            if len(test) >= 2: ndev = min(max(ndev, 1), len(test)-1)
            for j, i in enumerate(order): result.append({**test[i], "partition": "development" if j < ndev else "final"})
    return sorted(result, key=lambda r: r["id"])

def prepare(data_root, cfg):
    out = init_run("00_prepare", cfg); rows = []; excluded = []
    for scene in cfg["scenes"]:
        for split in ["training", "testing"]:
            for folder in sorted((Path(data_root)/scene/split/"frames").iterdir(), key=lambda p: int(p.name)):
                files = sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem))
                ids = [int(p.stem) for p in files]
                if ids != list(range(len(files))): raise ValueError(f"Noncontiguous source frames: {folder}")
                row = {"id": f"{scene}/{split}/{folder.name}", "scene": scene, "source_split": split, "video": folder.name,
                       "frames": len(files), "frame_listing_sha256": digest([(p.name, p.stat().st_size) for p in files]),
                       "valid": True, "stratum": "normal", "anomaly_frames": 0}
                if split == "testing":
                    labels = [p for p in (Path(data_root)/scene/"test_label").glob("*.npy") if int(p.stem) == int(folder.name)]
                    if len(labels) != 1: raise ValueError(f"Ambiguous/missing label: {row['id']}")
                    y = np.load(labels[0], allow_pickle=False)
                    if y.ndim != 1 or not np.isin(y, [0,1]).all(): raise ValueError("Invalid binary labels")
                    row.update(label_file=str(labels[0].relative_to(data_root)), label_sha256=file_hash(labels[0]), label_frames=len(y), anomaly_frames=int(y.sum()))
                    row["stratum"] = "normal" if not y.any() else "anomaly" if y.all() else "mixed"
                    if len(y) != len(files):
                        row["valid"] = False
                        excluded.append({"id": row["id"], "frames":len(files), "label_frames":len(y), "reason":"unresolved_frame_label_length_mismatch"})
                rows.append(row)
    manifest = partition(rows, cfg)
    write_json(out/"dataset_manifest.json", {"videos": rows, "identity_method": "ordered frame filenames/sizes and full label SHA256; not full image-content hash"})
    write_json(out/"splits.json", {"videos": manifest, "split_seed": cfg["split_seed"], "original_recording_groups": "unknown", "cycle_boundaries": "weak_recording_alignment", "manifest_sha256": digest(manifest)})
    pd.DataFrame(excluded, columns=["id", "frames", "label_frames", "reason"]).to_csv(out/"exclusions.csv", index=False)
    summary = pd.DataFrame(manifest).groupby(["scene", "partition"], sort=False).agg(videos=("id", "size"), frames=("frames", "sum")).reset_index()
    summary.to_csv(out/"data_counts.csv", index=False)
    complete_run(out, manifest_sha256=digest(manifest), videos=len(rows), eligible_videos=len(manifest), excluded_videos=len(excluded))
    return out

def load_rows(scene=None, parts=None):
    rows = read_json(ROOT/"results/00_prepare/splits.json")["videos"]
    return [r for r in rows if (scene is None or r["scene"] == scene) and (parts is None or r["partition"] in parts)]

def labels_for(root, row):
    if row["source_split"] == "training": return np.zeros(row["frames"], dtype=np.uint8)
    path=Path(root)/row["label_file"]
    if file_hash(path) != row["label_sha256"]: raise ValueError("Labels changed after split freeze")
    y=np.load(path, allow_pickle=False).astype(np.uint8)
    if len(y)!=row["frames"]: raise ValueError("Frame-label mismatch")
    return y
