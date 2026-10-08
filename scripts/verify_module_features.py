"""Small integration smoke on cached normal features; no anomaly evaluation."""
import time
import joblib
import numpy as np
from ipad_experiment.io import ROOT,read_json,write_json,source_identity
from ipad_experiment.data import load_rows
from ipad_experiment.ablation import ContinuousModule,load_features,BRANCHES
from cycle_vad.model import descriptor
cfg=read_json(ROOT/'configs/experiment.json');scene='R01'
folder=ROOT/'artifacts/features/frozen'/scene
rows={part:load_rows(scene,[part])[:2 if part=='fit' else 1] for part in ['fit','validation','reference','threshold']}
assert all(r['source_split']=='training' for values in rows.values() for r in values)
data={part:[load_features(folder,r) for r in values] for part,values in rows.items()}
start=time.perf_counter();model=ContinuousModule(cfg['cycle'],42).fit(data['fit'],data['validation'])
model.calibrate(data['reference'],data['threshold'],joblib.load(folder/'calibration.joblib')['calibrator'],cfg['threshold_quantile'])
a=data['validation'][0];scores=model.score(a)
assert descriptor(a).shape[1]==3072
assert model.conditional.basis.shape==model.constant.basis.shape
for name in BRANCHES:
 assert len(scores[name])==len(a['indices']) and np.isfinite(scores[name]).all()
 if name.startswith('S+'):assert np.all(scores[name]>=scores['S'])
cut=len(a['indices'])//2;b={k:v.copy() for k,v in a.items()}
b['global'][cut:]+=2;b['patches'][cut:]-=2
future_changed=model.score(b)
maximum=0.
for name in BRANCHES:
 delta=np.max(np.abs(scores[name][:cut]-future_changed[name][:cut]));maximum=max(maximum,float(delta))
 np.testing.assert_allclose(scores[name][:cut],future_changed[name][:cut],rtol=1e-6,atol=1e-7)
write_json(ROOT/'results/02_lora/module_integration_smoke.json',{'status':'passed','performance_experiment':False,'final_partition_used':False,'backbone':'frozen','normal_only_inputs':{k:[r['id'] for r in v] for k,v in rows.items()},'descriptor_dimension':descriptor(a).shape[1],'tested_branches':BRANCHES,'conditional_rank':model.conditional.basis.shape[0],'constant_rank':model.constant.basis.shape[0],'future_perturbation_prefix_max_abs_error':maximum,'seconds':time.perf_counter()-start,'scope':'Real feature dimensions and numerical/causal execution only; this small fit is discarded and never used in performance evaluation.',**source_identity()})
print('Real normal-feature module smoke passed: 3072 dimensions, 12 branches, matched ranks and causal prefixes.')
