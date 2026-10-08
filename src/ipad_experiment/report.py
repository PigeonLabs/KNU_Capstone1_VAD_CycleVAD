"""Rebuild README figures/tables from saved experiment logs; no GPU required."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from .io import ROOT, read_json, write_json

COLORS={"S":"#285F91","S+C":"#B5682B","S+P":"#728244","S+C+P":"#A3507C","S+C0":"#70777E","S+C0+P":"#9A8555"}
plt.rcParams.update({"figure.dpi":140,"savefig.dpi":160,"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.18,"axes.axisbelow":True,"font.family":"DejaVu Sans"})

def table(df):
    def val(x):
        if isinstance(x,(float,np.floating)): return "—" if not np.isfinite(x) else f"{x:.3f}"
        return str(x)
    return "| "+" | ".join(df.columns)+" |\n| "+" | ".join(["---"]*len(df.columns))+" |\n"+"\n".join("| "+" | ".join(val(x) for x in row)+" |" for row in df.itertuples(index=False,name=None))

def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,bbox_inches="tight",facecolor="white");plt.close(fig)

def preparation():
    out=ROOT/"results/00_prepare";df=pd.read_csv(out/"data_counts.csv")
    parts=["fit","validation","reference","threshold","development","final"]
    labels=["Fit","Normal validation","Reference","Threshold","Development","Final (reserved)"]
    colors=["#285F91","#81A3BF","#B5682B","#D8A47B","#728244","#B2B6B9"]
    fig,ax=plt.subplots(figsize=(9,4.5));bottom=np.zeros(4)
    for part,label,color in zip(parts,labels,colors):
        values=df[df.partition==part].set_index("scene").reindex(["R01","R02","R03","R04"])["videos"].fillna(0).to_numpy()
        ax.bar(["R01","R02","R03","R04"],values,bottom=bottom,label=label,color=color,edgecolor="white")
        for i,v in enumerate(values):
            if v: ax.text(i,bottom[i]+v/2,str(int(v)),ha="center",va="center",color="white" if part in ["fit","reference","development"] else "#242424")
        bottom+=values
    ax.set(ylabel="Videos",title="IPAD R01–R04: fixed video-level split",ylim=(0,max(bottom)*1.13))
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.12),ncol=3,frameon=False)
    fig.text(.12,.01,"R02 testing 12/13/14 excluded: unresolved frame/label length mismatch.",fontsize=9)
    save(fig,out/"figures/data_split.png")
    summary=df.pivot(index="scene",columns="partition",values="videos").reindex(columns=parts).reset_index()
    note="# 준비 단계 결과\n\n"+table(summary)+"\n\n영상 단위 고정 분할입니다. 원본 녹화 그룹·실제 cycle 경계는 미확인입니다.\n\n![Data split](figures/data_split.png)\n"
    if (out/"gpu_parity.json").exists():
        q=read_json(out/"gpu_parity.json")
        note+=f"\n실제 사전학습 DINOv2 3프레임 공식 구현 대조: **{q['status']}**. 특징 최대 오차 {q['feature_max_abs_error']:.3g}, PCA projector 최대 오차 {q['pca_projector_max_abs_error']:.3g}, 점수 최대 오차 {q['score_max_abs_error']:.3g}.\n"
    (out/"report.md").write_text(note)
    return summary

def stage_figures(out):
    metrics_path=out/"metrics_by_scene.csv"
    if not metrics_path.exists(): return None
    df=pd.read_csv(metrics_path)
    mean=df.groupby(["partition","scene","branch"],sort=False)[["auroc","ap","f1","tpr","fpr"]].mean().reset_index()
    for part,part_df in mean.groupby("partition"):
        branches=part_df.branch.unique();scenes=["R01","R02","R03","R04"]
        fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
        width=.75/len(branches);x=np.arange(4)
        for j,metric in enumerate(["auroc","ap"]):
            for k,b in enumerate(branches):
                values=part_df[part_df.branch==b].set_index("scene").reindex(scenes)[metric].to_numpy()*100
                bars=axes[j].bar(x+(k-(len(branches)-1)/2)*width,values,width,label=b,color=COLORS.get(b,"#70777E"))
                if len(branches)<=2: axes[j].bar_label(bars,fmt="%.1f",padding=3,fontsize=9)
            axes[j].set(xticks=x,xticklabels=scenes,title=metric.upper(),ylim=(0,110),ylabel="Score (%)")
        axes[0].legend(frameon=False);fig.suptitle(f"{out.name}: {part} evaluation")
        save(fig,out/f"figures/{part}_performance.png")
    scores=[]
    for p in sorted((out/"scores").glob("*.csv.gz")): scores.append(pd.read_csv(p,dtype={"video":str}))
    if scores:
        data=pd.concat(scores,ignore_index=True)
        for part in data.partition.unique():
            fig,axes=plt.subplots(1,2,figsize=(10,4.5))
            for scene,sd in data[(data.partition==part)&(data.seed==data.seed.min())].groupby("scene"):
                if sd.gt.nunique()<2: continue
                fpr,tpr,_=roc_curve(sd.gt,sd.S);precision,recall,_=precision_recall_curve(sd.gt,sd.S)
                axes[0].plot(fpr,tpr,label=scene);axes[1].plot(recall,precision,label=scene)
            axes[0].plot([0,1],[0,1],"--",color="#858585",linewidth=1)
            axes[0].set(xlabel="False positive rate",ylabel="True positive rate",title="S: ROC")
            axes[1].set(xlabel="Recall",ylabel="Precision",title="S: precision–recall")
            for ax in axes:ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.legend(frameon=False)
            fig.suptitle(f"{part}: seed {data.seed.min()}");save(fig,out/f"figures/{part}_curves.png")
        # Deterministic example: first mixed-label video by scene/video, not best-performing.
        candidates=data[data.seed==data.seed.min()].groupby(["scene","video","partition"],sort=True)
        for (scene,video,part),sd in candidates:
            if sd.gt.nunique()!=2:continue
            fig,ax=plt.subplots(figsize=(10,3.8));sd=sd.sort_values("frame")
            for b in [x for x in ["S","C","P","S+C+P"] if x in sd]:ax.plot(sd.frame,sd[b],label=b,linewidth=1.1)
            ax.fill_between(sd.frame,0,1,where=sd.gt.astype(bool),transform=ax.get_xaxis_transform(),alpha=.14,color="#A3507C",label="GT anomaly")
            ax.set(xlabel="Source frame index",ylabel="Calibrated score",title=f"{scene}/{video} · {part} · first mixed-label example")
            ax.legend(frameon=False,ncol=5);save(fig,out/"figures/score_example.png");break
    return mean

def update_readme():
    prep=ROOT/"results/00_prepare";counts=pd.read_csv(prep/"data_counts.csv")
    stages=[("00_prepare","데이터·설정·구현 검증"),("01_baseline","Frozen SubspaceAD"),("02_lora","LoRA 학습·채택 판단"),("03_ablation","진행도 모듈 ablation")]
    status=[]
    for name,label in stages:
        p=ROOT/"results"/name/"run_manifest.json"
        value=read_json(p)["status"] if p.exists() else "not_started"
        status.append({"단계":label,"상태":{"completed":"완료","running":"진행 중","failed":"실패","not_started":"미실행"}.get(value,value),"결과":f"[상세](results/{name}/report.md)" if (ROOT/"results"/name/"report.md").exists() else "—"})
    text="""# IPAD Cycle-Aware SubspaceAD

**연속 공정 진행도에 따라 정상 평균을 바꾸고 잔차 subspace를 공유하는 모듈의 유용성**을 IPAD R01–R04에서 검증합니다.

실험 순서는 **Frozen SubspaceAD → 정상 영상 LoRA의 채택 판단 → 선택한 백본에서 C/P ablation**입니다. 각 단계의 코드·로그·결과·그림을 함께 게시합니다. 아직 실행하지 않은 결과나 성능 개선을 주장하지 않습니다.

## 진행 상태

"""+table(pd.DataFrame(status))+"""

## 방법과 평가 규약

- DINOv2-base 336px, 공식 중간층 `[-4,-5]` 평균, 정상 patch PCA 99% 설명분산, 공식 재구성 잔차·확대/블러·상위 1% 집계. 원 논문의 Giant/672px 벤치마크 수치 재현이 아닌 **IPAD 방법 적용 실험**입니다.
- S: SubspaceAD, C: 진행도 조건부 외형 점수, P: 정렬·innovation·진행 오차. 결합은 `max`, 가중치 1로 고정합니다.
- 정상 training을 fit/validation/reference/threshold = 55/15/15/15%로 영상 단위 분리합니다. 유효 testing은 약 40% 개발 / 60% 최종 평가입니다. 프레임 단위 random split을 사용하지 않습니다.
- 주 지표는 장면별 frame AUROC와 네 장면 macro 평균입니다. AP, 정상 q99 임계값의 F1/TPR/FPR와 이벤트 탐지도 보고합니다. 모든 평가 프레임을 stride 1로 관측합니다.
- LoRA 채택 기준: 개발셋 seed 42·43·44 평균 macro AUROC +1%p 이상 **및** 영상 단위 paired bootstrap 95% 구간 하한 > 0. 최종 결과를 보고 백본을 다시 선택하지 않습니다.
- 실제 cycle 경계와 원본 녹화 그룹이 없습니다. `weak_recording_alignment`를 사용하며 cycle 위치 정확도·원본 그룹 독립성을 주장하지 않습니다.
- R02 testing 12·13·14는 프레임/라벨 길이가 불일치하여 공통 제외합니다. 자세한 목록은 [exclusions.csv](results/00_prepare/exclusions.csv)를 참고하세요.
- 과거 테스트 결과 확인 이력이 있는 후속 실험입니다. 최종 분할은 이번 모델 선택에서 제외하지만 완전히 새로운 미관측 데이터로 표현하지 않습니다.

![고정 데이터 분할](results/00_prepare/figures/data_split.png)

설정: [experiment.json](configs/experiment.json) · [분할 manifest](results/00_prepare/splits.json) · [상세 규약](docs/EXPERIMENT_DESIGN.md) · [출처와 대응](docs/SOURCE_MAPPING.md)
"""
    for name,title in stages[1:]:
        out=ROOT/"results"/name;p=out/"metrics_by_scene.csv"
        text+=f"\n## {title}\n\n"
        if not p.exists():text+="미실행 또는 집계 전입니다.\n";continue
        df=pd.read_csv(p);aggregate=df.groupby(["partition","branch"])[["auroc","ap","f1","tpr","fpr"]].mean().reset_index()
        for col in ["auroc","ap","f1","tpr","fpr"]:aggregate[col]*=100
        text+="장면·seed 평균, 단위 %. 개발셋과 최종 평가셋을 구분합니다.\n\n"+table(aggregate)+"\n\n"
        scene_table=df.groupby(["partition","scene","branch"])[["auroc","ap"]].mean().reset_index();scene_table[["auroc","ap"]]*=100
        text+=table(scene_table)+"\n\n"
        for figure in sorted((out/"figures").glob("*.png")):text+=f"![{figure.stem}]({figure.relative_to(ROOT).as_posix()})\n\n"
        if (out/"interpretation.md").exists():text+=(out/"interpretation.md").read_text()+"\n"
        text+=f"[상세 분석·로그](results/{name}/report.md)\n"
    text+="""
## 재현

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python scripts/download_model.py
python -m ipad_experiment.pipeline prepare --data-root /path/to/IPAD_dataset
python -m pytest -q
python scripts/verify_gpu.py
python -m ipad_experiment.pipeline baseline --data-root /path/to/IPAD_dataset
python -m ipad_experiment.pipeline report
```

후속 단계 CLI는 `train-lora`, `select-backbone`, `ablation`입니다. 각 단계는 구현·검증 후 실행 상태를 갱신합니다. `report`는 저장된 점수·지표에서 그림과 README를 다시 생성하며 GPU가 필요하지 않습니다. [실행/분석 노트북](notebooks/Experiment.ipynb)을 함께 제공합니다.

## 결과 파일 정책

GitHub에는 코드·설정·분할·구조화 로그·지표·압축 프레임 점수 CSV·시각자료를 보관합니다. `results/<run_id>`의 manifest에 실행 코드 commit과 SHA256을 남깁니다. 원본 영상/프레임, 특징 캐시, 가중치와 체크포인트는 로컬 `artifacts/`에 보존합니다. 압축 CSV만으로 지표를 독립 재계산할 수 있습니다.

## 출처

- [SubspaceAD 논문](https://arxiv.org/abs/2602.23013), [공식 구현](https://github.com/CLendering/SubspaceAD). 고정 commit 및 라이선스는 [provenance](configs/provenance.json), [Apache-2.0](vendor/SUBSPACEAD_LICENSE)에 기록합니다.
- [IPAD 논문](https://arxiv.org/abs/2404.15033).
- 사용자 제공 `CycleVAD_Colab.ipynb`의 내장 구현: 정상 템플릿·causal tracker·Fourier 평균·공유 잔차 PCA·정상 tail 보정. 노트북 SHA256은 provenance에 기록합니다.
"""
    (ROOT/"README.md").write_text(text)

def report(run=None):
    preparation()
    for out in sorted((ROOT/"results").iterdir()):
        if not out.is_dir() or out.name=="00_prepare" or (run and out.name!=run):continue
        mean=stage_figures(out)
        if mean is not None:
            body=f"# {out.name}\n\n"+table(mean)+"\n\n"
            for p in sorted((out/"figures").glob("*.png")):body+=f"![{p.stem}]({p.relative_to(out)})\n\n"
            body+="상세: [장면별 지표](metrics_by_scene.csv), [영상별 지표](metrics_by_video.csv), [실행 설정](config.json), [manifest](run_manifest.json), `logs/`, `scores/`.\n"
            if (out/"interpretation.md").exists():body+="\n"+(out/"interpretation.md").read_text()
            (out/"report.md").write_text(body)
    update_readme()
