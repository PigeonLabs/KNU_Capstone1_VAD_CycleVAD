from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import torch
from cycle_vad.model import TailCalibrator
from .io import ROOT, read_json, write_json, init_run, complete_run, log, digest, file_hash
from .data import prepare, load_rows, labels_for
from .encoder import Encoder
from .subspace import Moments, Scorer
from .evaluation import summarize, save_scores

def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp.npz");np.savez_compressed(tmp,**arrays);tmp.replace(path)

def cached(path,signature):
    if not path.exists(): return None
    with np.load(path,allow_pickle=False) as f:
        if str(f["signature"])!=signature: return None
        return {k:f[k] for k in f.files}

def extract_video(encoder,data_root,row,cache_dir,log_path,scorer=None,moments=False):
    identity={"encoder":encoder.signature(row),"pca":digest({k:scorer.params[k].tolist() if isinstance(scorer.params[k],np.ndarray) else scorer.params[k] for k in ["mu","components"]}) if scorer else None,"moments":moments}
    signature=digest(identity);path=cache_dir/f"{row['source_split']}_{row['video']}.npz"
    previous=cached(path,signature)
    if previous is not None:
        log(log_path,event="cache_hit",video=row["id"])
        return previous
    start=time.perf_counter();stats=Moments(encoder.model.config.hidden_size,encoder.device) if moments else None
    globals_,patches,scores=[],[],[]
    for tokens,g,p in encoder.batches(data_root,row):
        if stats is not None: stats.update(tokens)
        if scorer is not None: scores.extend(scorer(tokens))
        globals_.append(g);patches.append(p)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    arrays={"global":np.concatenate(globals_),"patches":np.concatenate(patches),"indices":np.arange(row["frames"]),
            "frame_count":np.array(row["frames"]),"signature":np.array(signature),"raw_S":np.asarray(scores),
            "extraction_seconds":np.array(seconds),"peak_vram_bytes":np.array(torch.cuda.max_memory_allocated())}
    if stats is not None: arrays.update(stats.arrays())
    save_npz(path,**arrays)
    log(log_path,event="extracted",video=row["id"],frames=row["frames"],seconds=seconds,peak_vram_bytes=int(torch.cuda.max_memory_allocated()))
    return arrays

def baseline_scene(data_root,cfg,out,scene,seed,adapter=None,parts=("development",)):
    identity="frozen" if adapter is None else f"lora_seed{seed}_{file_hash(adapter)[:12]}"
    folder=ROOT/"artifacts/features"/identity/scene;folder.mkdir(parents=True,exist_ok=True)
    logfile=out/"logs"/f"{scene}_seed{seed}.jsonl"
    encoder=Encoder(cfg,adapter=adapter);torch.cuda.reset_peak_memory_stats()
    write_json(out/"encoders"/f"{scene}_seed{seed}.json",encoder.identity)
    combined=Moments(encoder.model.config.hidden_size,"cuda")
    for row in load_rows(scene,["fit"]):
        arrays=extract_video(encoder,data_root,row,folder,logfile,moments=True)
        combined.merge(int(arrays["n"]),arrays["mean"],arrays["m2"])
    start=time.perf_counter();params=combined.pca(cfg["baseline"]["pca_variance"]);pca_seconds=time.perf_counter()-start
    joblib.dump(params,folder/"pca.joblib")
    scorer=Scorer(params,image_size=cfg["encoder"]["image_size"])
    raw={};training_parts=["validation","reference","threshold"]
    for row in load_rows(scene,training_parts+list(parts)):
        raw[row["id"]]=extract_video(encoder,data_root,row,folder,logfile,scorer=scorer)
    ref=np.concatenate([raw[r["id"]]["raw_S"] for r in load_rows(scene,["reference"])])
    calibrator=TailCalibrator().fit(ref)
    threshold_values=np.concatenate([calibrator.score(raw[r["id"]]["raw_S"]) for r in load_rows(scene,["threshold"])])
    threshold=float(np.quantile(threshold_values,cfg["threshold_quantile"],method="higher"))
    joblib.dump({"calibrator":calibrator,"threshold":threshold,"encoder":encoder.identity},folder/"calibration.joblib")
    report={"scene":scene,"seed":seed,"encoder":identity,"pca_rank":params["k"],"retained_variance":params["retained_variance"],"patch_count":params["n"],
            "thresholds":{"S":threshold},"pca_eigendecomposition_seconds":pca_seconds,"peak_vram_bytes":int(torch.cuda.max_memory_allocated()),
            "normal_split_sha256":digest(load_rows(scene,["fit","validation","reference","threshold"]))}
    write_json(out/"fit_reports"/f"{scene}_seed{seed}.json",report)
    tables=[]
    for row in load_rows(scene,parts):
        a=raw[row["id"]];df=pd.DataFrame({"scene":scene,"seed":seed,"video":row["video"],"partition":row["partition"],"frame":a["indices"],"gt":labels_for(data_root,row),
                                        "raw_S":a["raw_S"],"S":calibrator.score(a["raw_S"])})
        save_scores(out/"scores"/f"{scene}_seed{seed}_{row['partition']}_{row['video']}.csv.gz",df)
        tables.append(df)
    del encoder,scorer,combined;torch.cuda.empty_cache()
    return pd.concat(tables,ignore_index=True),{"S":threshold}

def baseline(data_root,cfg,scenes=None):
    out=init_run("01_baseline",cfg);tables={};thresholds={}
    for scene in scenes or cfg["scenes"]:
        df,t=baseline_scene(data_root,cfg,out,scene,42)
        tables[(scene,42)]=df;thresholds[(scene,42)]=t
    # Frozen extraction/PCA is deterministic: only one measured run, not three independent replications.
    videos,metrics=summarize(tables,thresholds,["S"])
    videos.to_csv(out/"metrics_by_video.csv",index=False);metrics.to_csv(out/"metrics_by_scene.csv",index=False)
    if set(scenes or cfg["scenes"])==set(cfg["scenes"]):
        complete_run(out,selection_partition="development",frozen_replication="one deterministic run shared by paired seeds")
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["prepare","baseline","train-lora","select-backbone","ablation","report"])
    parser.add_argument("--data-root",type=Path,default=ROOT.parent/"IPAD_dataset")
    parser.add_argument("--config",type=Path,default=ROOT/"configs/experiment.json")
    parser.add_argument("--scenes",nargs="+")
    parser.add_argument("--run",default=None)
    args=parser.parse_args();cfg=read_json(args.config)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.stage=="prepare": prepare(args.data_root,cfg)
    elif args.stage=="baseline": baseline(args.data_root,cfg,args.scenes)
    elif args.stage=="report":
        from .report import report
        report(args.run)
    elif args.stage=="train-lora":
        from .lora import run_training
        run_training(args.data_root,cfg,args.scenes)
    elif args.stage=="select-backbone":
        from .selection import select_backbone
        select_backbone(args.data_root,cfg)
    elif args.stage=="ablation":
        from .ablation import run_ablation
        run_ablation(args.data_root,cfg)

if __name__=="__main__": main()
