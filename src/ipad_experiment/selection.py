"""Paired cluster bootstrap and one prespecified global backbone decision."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .io import ROOT,init_run,complete_run,write_json,read_json,file_hash,log
from .data import load_rows
from .evaluation import summarize

def auc_components(df,score):
    """Exact Mann–Whitney numerator by positive-video / negative-video pair."""
    groups=[g.sort_values("frame") for _,g in df.groupby("video",sort=True)]
    ids=[str(g.video.iloc[0]) for g in groups]
    positives=[g.loc[g["gt"]==1,score].to_numpy() for g in groups]
    negatives=[np.sort(g.loc[g["gt"]==0,score].to_numpy()) for g in groups]
    matrix=np.zeros((len(ids),len(ids)))
    for i,pos in enumerate(positives):
        for j,neg in enumerate(negatives):
            matrix[i,j]=(np.searchsorted(neg,pos,side="left")+.5*(np.searchsorted(neg,pos,side="right")-np.searchsorted(neg,pos,side="left"))).sum()
    return ids,matrix,np.array([len(p) for p in positives]),np.array([len(n) for n in negatives])

def paired_bootstrap(reference,candidate,reference_score,candidate_score,split_rows,samples=10000,seed=20261009):
    """Dict[(scene,seed)] of identically aligned score tables; resample videos, not frames."""
    rng=np.random.default_rng(seed);scene_results=[];distributions=[]
    for scene in sorted({k[0] for k in candidate}):
        matrices=[];seed_deltas=[]
        for key in sorted(k for k in candidate if k[0]==scene):
            ref=reference.get(key,reference.get((scene,42)))
            a=ref.sort_values(["video","frame"]).reset_index(drop=True)
            b=candidate[key].sort_values(["video","frame"]).reset_index(drop=True)
            if not np.array_equal(a[["video","frame","gt"]].to_numpy(),b[["video","frame","gt"]].to_numpy()):raise ValueError("Unpaired evaluation frames")
            ids,u0,p0,n0=auc_components(a,reference_score);ids1,u1,p,n=auc_components(b,candidate_score)
            if ids!=ids1 or not np.array_equal(p,p0) or not np.array_equal(n,n0):raise ValueError("Unpaired labels")
            matrices.append(u1-u0);seed_deltas.append(float((u1-u0).sum()/(p.sum()*n.sum())))
        delta=np.mean(matrices,axis=0)
        weights=np.zeros((samples,len(ids)))
        row_map={r["video"]:r for r in split_rows if r["scene"]==scene}
        for stratum in ["normal","mixed","anomaly"]:
            indices=[i for i,v in enumerate(ids) if row_map[v]["stratum"]==stratum]
            if not indices:continue
            weights[:,indices]=rng.multinomial(len(indices),np.full(len(indices),1/len(indices)),size=samples)
        denominator=(weights@p)*(weights@n)
        if np.any(denominator==0):raise ValueError("Bootstrap stratum lacks both frame classes")
        differences=np.einsum("bi,ij,bj->b",weights,delta,weights)/denominator
        distributions.append(differences)
        scene_results.append({"scene":scene,"mean_delta":float(np.mean(seed_deltas)),"ci_low":float(np.quantile(differences,.025)),"ci_high":float(np.quantile(differences,.975)),"seed_deltas":seed_deltas,"videos":len(ids)})
    macro=np.mean(distributions,axis=0)
    result={"mean_delta":float(np.mean([r["mean_delta"] for r in scene_results])),"ci_low":float(np.quantile(macro,.025)),"ci_high":float(np.quantile(macro,.975)),"scene_results":scene_results,"bootstrap_samples":samples,"bootstrap_seed":seed,"unit":"video within scene/label stratum; paired across methods and seeds","caveat":"Unknown original recording groups; very small strata constrain uncertainty estimates."}
    return result,macro

def load_score_tables(out,partition="development"):
    groups={}
    for path in sorted((out/"scores").glob("*.csv.gz")):
        df=pd.read_csv(path,dtype={"video":str})
        df=df[df.partition==partition]
        if not len(df):continue
        key=(str(df.scene.iloc[0]),int(df.seed.iloc[0]));groups.setdefault(key,[]).append(df)
    return {k:pd.concat(v,ignore_index=True) for k,v in groups.items()}

def select_backbone(data_root,cfg):
    from .pipeline import baseline_scene
    out=init_run("02_lora",cfg);candidate={};thresholds={}
    for scene in cfg["scenes"]:
        for seed in cfg["seeds"]:
            report=out/"training"/f"{scene}_seed{seed}.json"
            adapter=ROOT/"artifacts/lora"/scene/f"seed{seed}"/"best.pt"
            if not report.exists() or read_json(report).get("status")!="completed":raise RuntimeError("All 12 normal fits must finish before selection")
            if read_json(report)["checkpoint_sha256"]!=file_hash(adapter):raise RuntimeError("Adapter changed")
            candidate[(scene,seed)],thresholds[(scene,seed)]=baseline_scene(data_root,cfg,out,scene,seed,adapter=adapter)
    videos,metrics=summarize(candidate,thresholds,["S"]);videos.to_csv(out/"metrics_by_video.csv",index=False);metrics.to_csv(out/"metrics_by_scene.csv",index=False)
    reference=load_score_tables(ROOT/"results/01_baseline")
    result,distribution=paired_bootstrap(reference,candidate,"S","S",load_rows(parts=["development"]),cfg["bootstrap"]["samples"],cfg["bootstrap"]["seed"])
    passed=result["mean_delta"]>=cfg["lora_adoption"]["macro_auroc_min_delta"] and result["ci_low"]>0
    result.update(selected_backbone="lora" if passed else "frozen",adopt_lora=passed,criteria=cfg["lora_adoption"],selection_partition="development",no_final_results_used=True)
    write_json(out/"backbone_decision.json",result)
    pd.DataFrame(result["scene_results"]).to_csv(out/"comparisons.csv",index=False)
    pd.DataFrame({"macro_auroc_delta":distribution}).to_csv(out/"bootstrap_deltas.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    complete_run(out,selected_backbone=result["selected_backbone"])
    return result
