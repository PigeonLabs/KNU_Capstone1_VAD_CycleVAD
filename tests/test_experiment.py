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
