import numpy as np
import pytest
import torch
from transformers import Dinov2Config, Dinov2Model
from ipad_experiment.data import allocations, partition
from ipad_experiment.encoder import install_lora, adapter_state
from ipad_experiment.subspace import Moments, Scorer
from ipad_experiment.evaluation import measure
from subspacead.core.pca import PCAModel
from subspacead.post_process.scoring import calculate_anomaly_scores, post_process_map, aggregate_image_score

def test_streaming_statistics_and_scores_match_official():
    rng=np.random.default_rng(8);x=rng.normal(size=(350,20)).astype(np.float32)
    x[:,10:]=x[:,:10]@rng.normal(size=(10,10)).astype(np.float32)+.1*x[:,10:]
    stats=Moments(20,"cpu")
    for batch in np.array_split(x,7):stats.update(torch.from_numpy(batch))
    actual=stats.pca(.99)
    official=PCAModel(ev=.99);official.device=torch.device("cpu")
    ref=official.fit(lambda:iter(np.array_split(x,7)),20,len(x),7)
    assert actual["k"]==ref["k"]
    np.testing.assert_allclose(actual["mu"],ref["mu"],atol=1e-12)
    np.testing.assert_allclose(actual["components"]@actual["components"].T,ref["components"]@ref["components"].T,atol=1e-11)
    test=rng.normal(size=(3,16,20)).astype(np.float32)
    ref_patch=calculate_anomaly_scores(test.reshape(-1,20),ref,"reconstruction").reshape(3,4,4)
    ref_score=np.array([aggregate_image_score(post_process_map(m,336),"mtop1p") for m in ref_patch])
    with torch.no_grad(): score=Scorer(actual,"cpu")(torch.from_numpy(test))
    np.testing.assert_allclose(score,ref_score,rtol=2e-6,atol=2e-6)

def test_zero_adapter_preserves_model_and_only_adapters_update():
    torch.manual_seed(42)
    model=Dinov2Model(Dinov2Config(hidden_size=24,num_hidden_layers=2,num_attention_heads=3,intermediate_size=48,image_size=28,patch_size=14)).eval()
    x=torch.randn(2,3,28,28)
    with torch.no_grad(): expected=model(x).last_hidden_state.clone()
    install_lora(model,{"targets":["query","value"],"rank":4,"alpha":8})
    frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
    assert all(n.endswith((".a",".b")) for n,p in model.named_parameters() if p.requires_grad)
    np.testing.assert_array_equal(expected.numpy(),model(x).last_hidden_state.detach().numpy())
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    model(x).last_hidden_state[:,:,0].square().mean().backward();optimizer.step()
    assert any(v.abs().sum()>0 for k,v in adapter_state(model).items() if k.endswith(".b"))
    for n,p in model.named_parameters():
        if n in frozen: assert torch.equal(p,frozen[n])

def test_split_is_disjoint_and_stratified():
    cfg={"scenes":["R01"],"split_seed":42,"normal_fractions":[.55,.15,.15,.15],"development_fraction":.4}
    rows=[]
    for split,n in [("training",22),("testing",12)]:
        for i in range(n):rows.append({"id":f"{split}/{i}","scene":"R01","source_split":split,"video":str(i),"valid":True,"stratum":"normal" if i<3 else "mixed"})
    parts=partition(rows,cfg)
    assert len(parts)==len({r['id'] for r in parts})==len(rows)
    assert partition(rows[::-1],cfg)==parts
    assert set(r["partition"] for r in parts if r["source_split"]=="training")=={"fit","validation","reference","threshold"}
    for part in ["development","final"]:
        assert {r["stratum"] for r in parts if r["partition"]==part}=={"normal","mixed"}
    assert sum(allocations(22,cfg["normal_fractions"]))==22

def test_metric_events_and_one_class_auroc():
    y=np.array([0,1,1,0,1,1,1]);s=np.array([0,0,2,0,0,0,2])
    m=measure(y,s,1)
    assert m["events"]==2 and m["detected_events"]==2 and m["mean_detected_event_delay_frames"]==1.5
    assert np.isnan(measure(np.ones(7),s,1)["auroc"])

def test_saved_split_has_no_invalid_rows():
    from ipad_experiment.data import load_rows
    rows=load_rows()
    assert not any(r["id"] in ["R02/testing/12","R02/testing/13","R02/testing/14"] for r in rows)
    assert all(r["valid"] for r in rows)

def test_video_pair_auc_equals_explicit_resampled_auc():
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from ipad_experiment.selection import auc_components,paired_bootstrap
    df=pd.DataFrame({"video":["01"]*4+["02"]*4,"frame":list(range(4))*2,"gt":[0,1,0,1,0,1,1,0],"S":[.1,.8,.8,.5,.6,.6,.9,.3]})
    ids,u,p,n=auc_components(df,"S");weights=np.array([3,2])
    expanded=pd.concat([df[df.video==v] for v,w in zip(ids,weights) for _ in range(w)])
    np.testing.assert_allclose(weights@u@weights/((weights@p)*(weights@n)),roc_auc_score(expanded["gt"],expanded.S))
    candidate=df.copy();candidate['S']=candidate["gt"]
    result,dist=paired_bootstrap({('R01',42):df},{('R01',42):candidate},'S','S',[{'scene':'R01','video':v,'stratum':'mixed'} for v in ids],samples=200,seed=42)
    np.testing.assert_allclose(result['mean_delta'],1-roc_auc_score(df["gt"],df.S))
    assert np.all(dist>=0)

def test_cycle_ablation_causality_rank_and_baseline_fallback():
    from test_cycle_vad import synthetic
    from ipad_experiment.ablation import ContinuousModule,BRANCHES
    from cycle_vad.model import TailCalibrator
    cfg={'cycle_grid':32,'cycle_latent':6,'harmonics_candidates':[2],'ridge_candidates':[.1],'fit_samples':200,'subspace_rank':6,'residual_variance':.95}
    def sample(seed):
        d=synthetic(seed);d['raw_S']=np.linspace(0,1,64);return d
    fit=[sample(i) for i in range(4)];model=ContinuousModule(cfg,42).fit(fit,[sample(5)])
    model.calibrate([sample(6)],[sample(7)],TailCalibrator().fit(np.linspace(0,1,256)),.99)
    assert model.conditional.basis.shape==model.constant.basis.shape
    a=sample(8);b={k:v.copy() for k,v in a.items()};b['global'][40:]+=2;b['patches'][40:]+=2
    sa,sb=model.score(a),model.score(b)
    for key in BRANCHES:np.testing.assert_allclose(sa[key][:40],sb[key][:40],rtol=1e-6,atol=1e-7)
    old_predict=model.tracker.predict
    def zero_conf(z,ids):
        out=old_predict(z,ids);out['confidence'][:]=0;return out
    model.tracker.predict=zero_conf
    s=model.score(a);np.testing.assert_array_equal(s['S'],s['S+C'])

def test_teacher_distillation_keeps_gradient_on_student_only():
    from ipad_experiment.lora import distill_loss
    target=[torch.randn(4,8,10)]
    prediction=[torch.randn(4,8,10,requires_grad=True)]
    loss=distill_loss(prediction,target);loss.backward()
    assert torch.isfinite(prediction[0].grad).all()
    assert target[0].grad is None

def test_summary_uses_gt_column_and_keeps_video_event_boundaries():
    import pandas as pd
    from ipad_experiment.evaluation import summarize
    df=pd.DataFrame({'video':['01','01','02','02'],'partition':'development','gt':[0,1,1,0],'S':[0,2,2,0]})
    videos,scenes=summarize({('R01',42):df},{('R01',42):{'S':1}},['S'])
    assert len(videos)==2 and scenes.iloc[0]['events']==2
    assert scenes.iloc[0]['auroc']==1 and scenes.iloc[0]['event_coverage']==1


def test_score_csv_roundtrip_preserves_threshold_equality(tmp_path):
    import pandas as pd
    from ipad_experiment.evaluation import save_scores
    threshold=float(-np.log(23/2349))
    values=np.array([np.nextafter(threshold,-np.inf),threshold,np.nextafter(threshold,np.inf)])
    path=tmp_path/'scores.csv.gz'
    save_scores(path,pd.DataFrame({'S':values,'gt':[0,1,1]}))
    actual=pd.read_csv(path,float_precision='round_trip').S.to_numpy()
    np.testing.assert_array_equal(actual,values)
    np.testing.assert_array_equal(actual>=threshold,[False,True,True])
