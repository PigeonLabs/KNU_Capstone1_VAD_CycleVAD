"""Prespecified conditional-mean and process ablations of the notebook module."""
from __future__ import annotations
import time
import joblib
import numpy as np
import pandas as pd
from cycle_vad.data import balanced_sample
from cycle_vad.model import descriptor,ResidualSpace,TailCalibrator
from cycle_vad.tracker import CycleTracker,fourier
from .io import ROOT,read_json,write_json,init_run,complete_run,log,file_hash,digest
from .data import load_rows,labels_for
from .evaluation import summarize,save_scores
from .selection import paired_bootstrap,load_score_tables

BRANCHES=["S","S+C","S+P","S+C+P","S+C0","S+C0+P","C","P","S+C_no_conf","S+alignment","S+innovation","S+progress"]
CONTRASTS=[["S","S+C+P"],["S","S+C"],["S","S+P"],["S+P","S+C+P"],["S+C","S+C+P"],["S+C0","S+C"],["S+C0+P","S+C+P"]]

class ContinuousModule:
    def __init__(self,cfg,seed): self.cfg=dict(cfg);self.seed=seed
    def fit(self,fit,validation):
        c=self.cfg;z=[descriptor(d) for d in fit];vz=[descriptor(d) for d in validation]
        self.tracker=CycleTracker(c["cycle_grid"],c["cycle_latent"],self.seed)
        segments=[[np.arange(len(x))] for x in z]
        self.tracker.fit(z,[d["indices"] for d in fit],segments)
        cycles=[self.tracker.predict(x,d["indices"]) for x,d in zip(z,fit)]
        vcycles=[self.tracker.predict(x,d["indices"]) for x,d in zip(vz,validation)]
        choices=[];best=None
        for h in c["harmonics_candidates"]:
            for ridge in c["ridge_candidates"]:
                design=[fourier(t["angle"],h) for t in cycles]
                rows=balanced_sample([np.concatenate([b,x,np.maximum(t["confidence"],.1)[:,None]],axis=1).astype(np.float32) for b,x,t in zip(design,z,cycles)],c["fit_samples"],self.seed)
                cols=1+2*h;b,x,w=rows[:,:cols],rows[:,cols:-1],rows[:,-1]
                penalty=np.eye(cols)*ridge*len(b);penalty[0,0]=1e-8
                coef=np.linalg.solve(b.T@(w[:,None]*b)+penalty,b.T@(w[:,None]*x)).astype(np.float32)
                loss=float(np.mean([np.mean((x-fourier(t["angle"],h)@coef)**2) for x,t in zip(vz,vcycles)]))
                choices.append({"harmonics":h,"ridge":ridge,"normal_validation_mse":loss})
                if best is None or loss<best[0]:best=(loss,h,ridge,coef)
        _,self.harmonics,self.ridge,self.coef=best
        residuals=[x-fourier(t["angle"],self.harmonics)@self.coef for x,t in zip(z,cycles)]
        self.conditional=ResidualSpace().fit(balanced_sample(residuals,c["fit_samples"],self.seed),c["subspace_rank"],variance=c["residual_variance"],seed=self.seed)
        rows=balanced_sample([np.concatenate([x,np.maximum(t["confidence"],.1)[:,None]],axis=1).astype(np.float32) for x,t in zip(z,cycles)],c["fit_samples"],self.seed)
        self.constant_mean=np.average(rows[:,:-1],axis=0,weights=rows[:,-1]).astype(np.float32)
        rank=self.conditional.basis.shape[0]
        self.constant=ResidualSpace().fit(balanced_sample([x-self.constant_mean for x in z],c["fit_samples"],self.seed),rank,variance=1.,seed=self.seed)
        assert self.constant.basis.shape[0]==rank
        self.diagnostics={"seed":self.seed,"descriptor_dimension":z[0].shape[1],"cycle":self.tracker.fit_diagnostics,"conditional":self.conditional.diagnostics,"constant":self.constant.diagnostics,"selection":choices,"harmonics":self.harmonics,"ridge":self.ridge,"mean_fit_confidence":float(np.mean([t["confidence"].mean() for t in cycles])),"boundary_source":"weak_recording_alignment"}
        return self

    def raw(self,data):
        z=descriptor(data);t=self.tracker.predict(z,data["indices"])
        outside,inside,_=self.conditional.score(z-fourier(t["angle"],self.harmonics)@self.coef)
        outside0,inside0,_=self.constant.score(z-self.constant_mean)
        raw={"conditional":outside,"conditional_inside":inside,"constant":outside0,"constant_inside":inside0,**{k:t[k] for k in ["alignment","innovation","progress"]}}
        return raw,t

    def calibrate(self,reference,threshold,s_calibrator,q):
        raw=[self.raw(d)[0] for d in reference]
        self.calibrators={k:TailCalibrator().fit(np.concatenate([r[k] for r in raw])) for k in raw[0]}
        self.s_calibrator=s_calibrator
        scores=[self.score(d) for d in threshold]
        self.thresholds={b:float(np.quantile(np.concatenate([s[b] for s in scores]),q,method="higher")) for b in BRANCHES}
        return self

    def score(self,data):
        raw,t=self.raw(data);c={k:self.calibrators[k].score(v) for k,v in raw.items()}
        s=self.s_calibrator.score(data["raw_S"])
        no_conf=np.maximum(c["conditional"],c["conditional_inside"])
        conditioned=t["confidence"]*no_conf
        constant=t["confidence"]*np.maximum(c["constant"],c["constant_inside"])
        process=np.maximum.reduce([c[k] for k in ["alignment","innovation","progress"]])
        return {"S":s,"C":conditioned,"C0":constant,"P":process,"S+C":np.maximum(s,conditioned),"S+P":np.maximum(s,process),"S+C+P":np.maximum.reduce([s,conditioned,process]),
                "S+C0":np.maximum(s,constant),"S+C0+P":np.maximum.reduce([s,constant,process]),"S+C_no_conf":np.maximum(s,no_conf),
                **{f"S+{k}":np.maximum(s,c[k]) for k in ["alignment","innovation","progress"]},"angle":t["angle"],"confidence":t["confidence"],
                **{f"raw_{k}":v for k,v in raw.items()}}

def load_features(folder,row):
    p=folder/f"{row['source_split']}_{row['video']}.npz"
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}

def run_ablation(data_root,cfg):
    from .pipeline import baseline_scene
    decision_path=ROOT/"results/02_lora/backbone_decision.json"
    if not decision_path.exists():raise RuntimeError("Backbone selection must finish before final evaluation")
    decision=read_json(decision_path)
    lora_out=decision_path.parent
    if read_json(lora_out/"run_manifest.json")["status"]!="completed":raise RuntimeError("Backbone selection is incomplete")
    if digest(read_json(lora_out/"config.json"))!=digest(cfg):raise ValueError("Final configuration differs from selection")
    out=init_run("03_ablation",cfg)
    frozen_protocol={"backbone_decision_sha256":file_hash(decision_path),"branches":BRANCHES,"contrasts":CONTRASTS,"config_sha256":digest(cfg),"partition":"final","no_final_selection":True}
    protocol_path=out/"frozen_protocol.json"
    if protocol_path.exists() and read_json(protocol_path)!=frozen_protocol:raise ValueError("Cannot change the frozen final protocol")
    write_json(protocol_path,frozen_protocol)
    all_tables={};thresholds={};frozen_tables={};frozen_thresholds={};timings=[]
    for scene in cfg["scenes"]:
        # Final original baseline is evaluated once, after the adoption decision is frozen.
        frozen_tables[(scene,42)],frozen_thresholds[(scene,42)]=baseline_scene(data_root,cfg,out/"frozen_baseline",scene,42,parts=("final",))
        for seed in cfg["seeds"]:
            adapter=ROOT/"artifacts/lora"/scene/f"seed{seed}"/"best.pt" if decision["adopt_lora"] else None
            identity="frozen" if adapter is None else f"lora_seed{seed}_{file_hash(adapter)[:12]}"
            if adapter is not None:baseline_scene(data_root,cfg,out/"selected_baseline",scene,seed,adapter,parts=("final",))
            folder=ROOT/"artifacts/features"/identity/scene
            groups={part:[load_features(folder,r) for r in load_rows(scene,[part])] for part in ["fit","validation","reference","threshold"]}
            start=time.perf_counter();model=ContinuousModule(cfg["cycle"],seed).fit(groups["fit"],groups["validation"])
            s_calibration=joblib.load(folder/"calibration.joblib")
            model.calibrate(groups["reference"],groups["threshold"],s_calibration["calibrator"],cfg["threshold_quantile"])
            fit_seconds=time.perf_counter()-start
            write_json(out/"fit_reports"/f"{scene}_seed{seed}.json",{**model.diagnostics,"thresholds":model.thresholds,"fit_calibrate_seconds":fit_seconds,"backbone":identity})
            checkpoint=ROOT/"artifacts/cycle"/scene/f"seed{seed}.joblib";checkpoint.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,checkpoint)
            frames=[]
            for row in load_rows(scene,["final"]):
                features=load_features(folder,row);start=time.perf_counter();scores=model.score(features);duration=time.perf_counter()-start
                df=pd.DataFrame({"scene":scene,"seed":seed,"video":row["video"],"partition":"final","frame":features["indices"],"gt":labels_for(data_root,row),"raw_S":features["raw_S"],**scores})
                save_scores(out/"scores"/f"{scene}_seed{seed}_final_{row['video']}.csv.gz",df);frames.append(df)
                timings.append({"scene":scene,"seed":seed,"video":row["video"],"frames":row["frames"],"cycle_score_seconds":duration})
            all_tables[(scene,seed)]=pd.concat(frames,ignore_index=True);thresholds[(scene,seed)]=model.thresholds
            log(out/"logs"/"cycle.jsonl",event="cycle_complete",scene=scene,seed=seed,fit_calibrate_seconds=fit_seconds,rank=model.conditional.basis.shape[0])
    videos,metrics=summarize(all_tables,thresholds,BRANCHES)
    videos.to_csv(out/"metrics_by_video.csv",index=False);metrics.to_csv(out/"metrics_by_scene.csv",index=False)
    fv,fm=summarize(frozen_tables,frozen_thresholds,["S"]);fv.to_csv(out/"frozen_baseline/metrics_by_video.csv",index=False);fm.to_csv(out/"frozen_baseline/metrics_by_scene.csv",index=False)
    pd.DataFrame(timings).to_csv(out/"timing.csv",index=False)
    comparisons=[]
    for a,b in CONTRASTS:
        result,distribution=paired_bootstrap(all_tables,all_tables,a,b,load_rows(parts=["final"]),cfg["bootstrap"]["samples"],cfg["bootstrap"]["seed"])
        name=f"{b}_minus_{a}";write_json(out/"comparisons"/f"{name}.json",result)
        comparisons.append({"contrast":name,**{k:result[k] for k in ["mean_delta","ci_low","ci_high"]}})
    pd.DataFrame(comparisons).to_csv(out/"comparisons.csv",index=False)
    complete_run(out,selected_backbone=decision["selected_backbone"],evaluation_partition="final")
    return out
