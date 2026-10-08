"""Metrics are recomputable from published scalar score tables."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def measure(y,scores,threshold):
    y=np.asarray(y,dtype=np.uint8); scores=np.asarray(scores,dtype=float)
    if len(y)!=len(scores) or not np.isfinite(scores).all(): raise ValueError("Invalid score/label alignment")
    pred=scores>=threshold; pos=y==1; neg=~pos
    auroc=float(roc_auc_score(y,scores)) if pos.any() and neg.any() else np.nan
    ap=float(average_precision_score(y,scores)) if pos.any() else np.nan
    starts=np.where(np.diff(np.r_[0,y,0].astype(int))==1)[0]
    ends=np.where(np.diff(np.r_[0,y,0].astype(int))==-1)[0]
    delays=[]
    for a,b in zip(starts,ends):
        hits=np.flatnonzero(pred[a:b])
        if len(hits): delays.append(int(hits[0]))
    return {"auroc":auroc,"ap":ap,"f1":float(f1_score(y,pred,zero_division=0)),
            "tpr":float(pred[pos].mean()) if pos.any() else np.nan,"fpr":float(pred[neg].mean()) if neg.any() else np.nan,
            "frames":len(y),"positive_frames":int(pos.sum()),"events":len(starts),"detected_events":len(delays),
            "event_coverage":len(delays)/len(starts) if len(starts) else np.nan,
            "delay_sum":sum(delays),"mean_detected_event_delay_frames":float(np.mean(delays)) if delays else np.nan}

def summarize(tables,thresholds,branches):
    video=[]; scene=[]
    for (scene_id,seed),frames in tables.items():
        for branch in branches:
            t=thresholds[(scene_id,seed)][branch]
            summaries=[]
            for (vid,part),df in frames.groupby(["video","partition"],sort=True):
                result=measure(df.gt,df[branch],t)
                record={"scene":scene_id,"seed":seed,"branch":branch,"video":str(vid),"partition":part,"threshold":t,**result}
                video.append(record); summaries.append(record)
            for part,df in frames.groupby("partition",sort=True):
                result=measure(df.gt,df[branch],t)
                sv=[r for r in summaries if r["partition"]==part]
                ne=sum(r["events"] for r in sv); nd=sum(r["detected_events"] for r in sv); delay=sum(r["delay_sum"] for r in sv)
                result.update(events=ne,detected_events=nd,delay_sum=delay,event_coverage=nd/ne if ne else np.nan,mean_detected_event_delay_frames=delay/nd if nd else np.nan)
                scene.append({"scene":scene_id,"seed":seed,"branch":branch,"partition":part,"threshold":t,**result})
    return pd.DataFrame(video),pd.DataFrame(scene)

def save_scores(path,df):
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False,float_format="%.12g",compression={"method":"gzip","mtime":0})
