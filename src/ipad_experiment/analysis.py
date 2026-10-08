"""Post-run evidence and figures derived only from saved logs and scores."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .io import ROOT, read_json, write_json

PRIMARY = ['S', 'S+C', 'S+P', 'S+C+P', 'S+C0', 'S+C0+P']
SCENES = ['R01', 'R02', 'R03', 'R04']


def uncertainty_figure(rows, path, title):
    from .report import save
    fig, ax = plt.subplots(figsize=(9, max(3.5, .52*len(rows)+1.2)))
    for i, row in enumerate(rows):
        point, lo, hi = [row[k]*100 for k in ['mean_delta', 'ci_low', 'ci_high']]
        color = '#285F91' if lo > 0 else '#70777E'
        ax.plot([lo, hi], [i, i], color=color, linewidth=2)
        ax.scatter(point, i, color=color, s=38, zorder=3)
        ax.annotate(f'{point:+.2f} [{lo:+.2f}, {hi:+.2f}]', (hi, i), xytext=(6, 0), textcoords='offset points', va='center', fontsize=9)
    ax.axvline(0, color='#252525', linewidth=1)
    ax.set(yticks=np.arange(len(rows)), yticklabels=[r['label'] for r in rows], xlabel='AUROC difference (percentage points) · paired video bootstrap 95% CI', title=title)
    ax.invert_yaxis();ax.margins(x=.45);save(fig, path)


def lora_analysis(out):
    from .report import save, table, SCENE_COLORS
    records = []
    for p in sorted((out/'training').glob('*.json')):
        d=read_json(p)
        if d['status']=='completed': records.append(d)
    if records:
        summary=pd.DataFrame([{k:d[k] for k in ['scene','seed','best_epoch','best_normal_validation_loss','epochs_completed','seconds','peak_vram_bytes','trainable_parameters']} for d in records])
        summary.to_csv(out/'training_summary.csv', index=False)
        fig, axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
        for scene,ax in zip(SCENES,axes.flat):
            for seed in [42,43,44]:
                p=out/'logs'/f'{scene}_seed{seed}_epochs.jsonl'
                if not p.exists():continue
                log=pd.DataFrame([json.loads(line) for line in p.read_text().splitlines()])
                ax.plot(log.epoch,log.validation_loss,label=f'Seed {seed}',linestyle={42:'-',43:'--',44:':'}[seed],color=SCENE_COLORS[scene])
            ax.set(title=scene,xlabel='Epoch (−1: untrained adapter)',ylabel='Normal validation relative MSE');ax.legend(frameon=False)
        fig.suptitle('LoRA checkpoint selection: normal validation only');fig.tight_layout();save(fig,out/'figures/normal_validation_loss.png')
    p=out/'backbone_decision.json'
    if not p.exists():return
    decision=read_json(p)
    uncertainty_figure([{'label':r['scene'],**r} for r in decision['scene_results']]+[{'label':'Macro (4 scenes)',**decision}],out/'figures/lora_vs_frozen.png','Development: LoRA minus frozen SubspaceAD')
    baseline=pd.read_csv(ROOT/'results/01_baseline/metrics_by_scene.csv',float_precision="round_trip").set_index('scene')
    measured=pd.read_csv(out/'metrics_by_scene.csv',float_precision="round_trip")
    lora=measured.groupby('scene')[['auroc','ap']].mean()
    per_seed=measured.groupby('seed')[['auroc','ap']].mean().reset_index()
    per_seed['delta_auroc_pp']=100*(per_seed.auroc-baseline.auroc.mean())
    per_seed.to_csv(out/'performance_by_seed.csv',index=False)
    seed_description=', '.join(f"seed {int(r.seed)}: {100*r.auroc:.2f}%" for r in per_seed.itertuples())
    comparison=pd.DataFrame({'frozen_auroc':baseline.auroc,'lora_auroc':lora.auroc,'delta_pp':100*(lora.auroc-baseline.auroc),'frozen_ap':baseline.ap,'lora_ap':lora.ap}).reset_index()
    comparison.to_csv(out/'frozen_comparison.csv',index=False)
    adopted=decision['adopt_lora']
    decision_text='LoRA를 채택합니다.' if adopted else 'LoRA를 채택하지 않고 frozen DINOv2로 모듈 실험을 진행합니다.'
    macro=100*lora.auroc.mean();frozen_macro=100*baseline.auroc.mean()
    text=f'''### 채택 판단

**{decision_text}** 개발셋 macro AUROC는 frozen **{frozen_macro:.2f}%**, LoRA seed 평균 **{macro:.2f}%**입니다. 차이는 **{decision['mean_delta']*100:+.2f}%p**, 영상 단위 paired bootstrap 95% 구간은 **[{decision['ci_low']*100:+.2f}, {decision['ci_high']*100:+.2f}]%p**입니다.

사전에 정한 두 조건(평균 +1%p 이상, 95% 하한 > 0)을 모두 만족할 때만 LoRA를 채택합니다. 정상 validation 손실 감소 자체를 이상 탐지 향상으로 해석하지 않습니다. 장면별로 유리한 백본을 따로 고르지 않았으며 최종 평가 라벨은 채택 판단에 사용하지 않았습니다.

[채택 근거 JSON](results/02_lora/backbone_decision.json) · [frozen 대조표](results/02_lora/frozen_comparison.csv) · [학습 요약](results/02_lora/training_summary.csv)

Seed별 macro AUROC는 **{seed_description}**입니다. [Seed별 성능표](results/02_lora/performance_by_seed.csv)에 AP와 frozen 대비 차이도 보존합니다. Bootstrap 구간은 관측된 세 seed 평균을 대상으로 영상 표집의 불확실성을 나타냅니다.

### 학습 및 불확실성

12개 fit 모두 정상 training/validation 영상만 사용했습니다. 원래 DINOv2 가중치의 학습 전후 SHA256 일치와 adapter 체크포인트 SHA256을 각 training 로그에 남겼습니다. 신뢰구간은 장면·라벨 유형 내 영상을 재표집하고 같은 표집을 모든 seed와 두 백본에 적용한 10,000회 결과입니다. 프레임이나 seed를 독립 표본으로 세지 않습니다. 원본 녹화 그룹을 알 수 없고 일부 strata의 영상 수가 작아 이 구간을 광범위한 일반화의 보장으로 해석할 수 없습니다.
'''
    if records:
        text+=f"\n전체 LoRA 학습 기록 합계 {sum(d['seconds'] for d in records)/60:.1f}분, 관측 peak allocated VRAM {max(d['peak_vram_bytes'] for d in records)/2**30:.2f}GiB입니다. 평가 추출 비용은 별도의 영상 로그에 있습니다.\n"
    (out/'interpretation.md').write_text(text)


def ablation_analysis(out):
    from .report import save, table
    path=out/'metrics_by_scene.csv'
    if not path.exists():return
    metrics=pd.read_csv(path,float_precision="round_trip");means=metrics.groupby(['scene','branch'])[['auroc','ap','tpr','fpr']].mean()
    matrix=means.auroc.unstack('scene').reindex(index=PRIMARY,columns=SCENES)*100
    matrix['Macro']=matrix.mean(axis=1)
    fig,ax=plt.subplots(figsize=(8,5))
    im=ax.imshow(matrix,aspect='auto',cmap='Blues',vmin=0,vmax=100)
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            v=matrix.iloc[i,j];ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if v>60 else '#252525')
    ax.grid(False);ax.set(xticks=range(5),xticklabels=matrix.columns,yticks=range(len(matrix)),yticklabels=matrix.index,title='Final evaluation: mean AUROC (%) over seeds 42/43/44')
    fig.colorbar(im,ax=ax,label='AUROC (%)');save(fig,out/'figures/final_primary_ablation.png')
    comparisons=pd.read_csv(out/'comparisons.csv',float_precision="round_trip")
    uncertainty_figure([{'label':r['contrast'].replace('_minus_',' − '),**r} for r in comparisons.to_dict('records')],out/'figures/ablation_differences.png','Final evaluation: prespecified component contrasts')
    diagnostics=[]
    tables=[]
    for p in sorted((out/'scores').glob('*.csv.gz')):
        d=pd.read_csv(p,dtype={'video':str},float_precision="round_trip");tables.append(d)
        for label,part in d.groupby('gt'):
            diagnostics.append({'scene':d.scene.iloc[0],'seed':int(d.seed.iloc[0]),'video':d.video.iloc[0],'gt':int(label),'frames':len(part),'confidence_mean':part.confidence.mean(),'confidence_q10':part.confidence.quantile(.1),'c_above_s_fraction':(part.C>part.S).mean(),'p_above_s_fraction':(part.P>part.S).mean(),'c_changes_sp_fraction':(part['S+C+P']>part['S+P']).mean()})
    pd.DataFrame(diagnostics).to_csv(out/'component_activity_by_video.csv',index=False)
    data=pd.concat(tables,ignore_index=True)
    # Diagnostic examples are explicitly the highest FPR and lowest TPR under S.
    video_metrics=pd.read_csv(out/'metrics_by_video.csv',dtype={'video':str},float_precision="round_trip")
    candidates=video_metrics[(video_metrics.seed==42)&(video_metrics.branch=='S')]
    selected=[]
    for metric,ascending,label in [('fpr',False,'Highest baseline FPR'),('tpr',True,'Lowest baseline TPR')]:
        available=candidates[candidates[metric].notna()].sort_values([metric,'scene','video'],ascending=[ascending,True,True])
        if available.empty:continue
        row=available.iloc[0];d=data[(data.seed==42)&(data.scene==row.scene)&(data.video==row.video)].sort_values('frame')
        fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True,gridspec_kw={'height_ratios':[3,1]})
        for b,color in [('S','#285F91'),('S+C+P','#A3507C')]:
            t=video_metrics[(video_metrics.scene==row.scene)&(video_metrics.seed==42)&(video_metrics.video==row.video)&(video_metrics.branch==b)].threshold.iloc[0]
            axes[0].plot(d.frame,d[b],label=b,color=color,linewidth=1)
            axes[0].axhline(t,color=color,linestyle='--',alpha=.65,label=f'{b} normal q99')
        axes[0].fill_between(d.frame,0,1,where=d['gt'].astype(bool),transform=axes[0].get_xaxis_transform(),alpha=.12,color='#A3507C',label='GT anomaly')
        axes[0].set(ylabel='Calibrated score',title=f'{label}: {row.scene}/{row.video} · seed 42');axes[0].legend(frameon=False,ncol=3,fontsize=8)
        axes[1].plot(d.frame,d.confidence,color='#728244');axes[1].set(xlabel='Source frame index',ylabel='Confidence',ylim=(0,1))
        fig.tight_layout();save(fig,out/f'figures/diagnostic_{metric}.png')
        selected.append({'scene':row.scene,'video':row.video,'seed':42,'selection':label,'baseline_value':row[metric]})
    write_json(out/'diagnostic_examples.json',{'rule':'Rank videos by baseline S failure metric; tie-break scene/video; no method selection from these examples.','examples':selected})
    frozen=pd.read_csv(out/'frozen_baseline/metrics_by_scene.csv',float_precision="round_trip")
    macro=matrix['Macro'];gain=macro['S+C+P']-macro['S']
    overall=comparisons[comparisons.contrast=='S+C+P_minus_S'].iloc[0]
    causal=comparisons[comparisons.contrast=='S+C+P_minus_S+P'].iloc[0]
    control=comparisons[comparisons.contrast=='S+C+P_minus_S+C0+P'].iloc[0]
    without_p=comparisons[comparisons.contrast=='S+C_minus_S'].iloc[0]
    control_without_p=comparisons[comparisons.contrast=='S+C_minus_S+C0'].iloc[0]
    if causal.ci_low>0 and control.ci_low>0:
        conclusion='S+P에 C를 추가한 최종 결합 맥락에서, C 추가와 상수 평균 대조가 모두 양의 구간을 보여 연속 정상 평균의 추가 기여를 지지합니다.'
    elif causal.ci_high<0 or control.ci_high<0:
        conclusion='S+P에 C를 추가한 최종 결합 맥락에서, 핵심 대조 중 음의 구간이 있어 연속 정상 평균의 유용성이 확인됐다고 결론 내릴 수 없습니다.'
    else:
        conclusion='S+P에 C를 추가한 최종 결합 맥락에서, 핵심 대조 중 0을 포함하는 구간이 있어 연속 정상 평균의 추가 기여를 확정할 근거는 부족합니다.'
    fits=[read_json(p) for p in sorted((out/'fit_reports').glob('*.json'))]
    fit_note=''
    if fits:
        ranks=[f['conditional']['rank'] for f in fits]
        retained=[f['conditional']['retained_variance'] for f in fits]
        reached=sum(f['conditional']['variance_target_met'] for f in fits)
        fit_note=f'공유 잔차 PCA의 실제 rank 범위는 {min(ranks)}–{max(ranks)}, 보존 설명분산 범위는 {100*min(retained):.2f}–{100*max(retained):.2f}%입니다. 95% 설명분산 목표를 달성한 fit은 {reached}/{len(fits)}개입니다. rank 상한 때문에 목표를 달성하지 못한 경우도 그대로 보고합니다.'
    text=f'''### 모듈 결과 해석

선택한 백본의 최종 평가 macro AUROC는 **S {macro['S']:.2f}% → S+C+P {macro['S+C+P']:.2f}%**, 차이는 **{gain:+.2f}%p**입니다. 별도로 측정한 원래 frozen S는 **{frozen.auroc.mean()*100:.2f}%**입니다. 개발셋 결과와 최종 평가 수치를 직접 증감 비교하지 않습니다.

전체 결합의 차이(S+C+P − S)에 대한 영상 단위 paired bootstrap 95% 구간은 **[{overall.ci_low*100:+.2f}, {overall.ci_high*100:+.2f}]%p**입니다.

P가 없는 경우, C를 S에 추가한 차이(S+C − S)는 **{without_p.mean_delta*100:+.2f}%p**, 95% 구간 **[{without_p.ci_low*100:+.2f}, {without_p.ci_high*100:+.2f}]%p**입니다. 상수 평균 대조(S+C − S+C0)는 **{control_without_p.mean_delta*100:+.2f}%p**, 95% 구간 **[{control_without_p.ci_low*100:+.2f}, {control_without_p.ci_high*100:+.2f}]%p**입니다. 이 결과와 아래의 P 포함 대조를 구분해 해석합니다.

연속 정상 평균 C를 S+P에 추가한 차이는 **{causal.mean_delta*100:+.2f}%p**, 영상 bootstrap 95% 구간은 **[{causal.ci_low*100:+.2f}, {causal.ci_high*100:+.2f}]%p**입니다. 같은 특징·표본·실제 rank·confidence를 사용하는 상수 평균 C0와의 비교(S+C+P − S+C0+P)는 **{control.mean_delta*100:+.2f}%p**, 95% 구간 **[{control.ci_low*100:+.2f}, {control.ci_high*100:+.2f}]%p**입니다.

**{conclusion}**

전체 결합의 향상만으로 연속 평균의 기여를 주장하지 않습니다. C 추가 및 C0 대조와 함께 판단하며, 95% 구간이 0을 포함하는 대조는 양의 효과가 확정됐다고 표현하지 않습니다. {len(comparisons)}개 대조의 구간은 다중 비교 보정 전 탐색적 구간입니다.

[대조별 수치](results/03_ablation/comparisons.csv) · [영상별 구성요소 활성도](results/03_ablation/component_activity_by_video.csv) · [실패 예시 선택 규칙](results/03_ablation/diagnostic_examples.json)

### 적용 범위와 진단

이 실험은 실제 cycle 경계 없이 정상 녹화 영상 전체를 cycle 후보로 삼는 `weak_recording_alignment` 조건입니다. 추정 angle/confidence는 정상 템플릿과의 정합도이며 정답 진행도와의 오차를 측정한 값이 아닙니다. 영상별 confidence와 C/P가 S를 바꾼 프레임 비율을 남겼습니다. 진단 그림은 baseline FPR이 가장 높은 영상과 TPR이 가장 낮은 영상을 고정 규칙으로 선택하며, 좋은 사례만 고르지 않습니다.

각 branch 임계값은 동일한 정상 threshold 영상의 q99로 따로 정했습니다. 최고 AUROC branch를 사후 선택하거나 최종 라벨로 임계값을 조정하지 않았습니다. rank·설명분산과 정상 validation 선택 내역은 `fit_reports/`, 이벤트 coverage·검출된 이벤트의 지연은 지표 CSV에 있습니다. 지연은 검출된 이벤트만의 조건부 값이므로 미검출 비율과 함께 읽어야 합니다.

{fit_note}
'''
    (out/'interpretation.md').write_text(text)


def build_analysis(out):
    if out.name=='02_lora':lora_analysis(out)
    elif out.name=='03_ablation':ablation_analysis(out)
