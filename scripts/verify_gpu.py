"""Check real pretrained features and official scoring before publishing baseline."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"vendor")]
import numpy as np
import torch
from PIL import Image
from ipad_experiment.encoder import Encoder
from ipad_experiment.io import read_json,write_json,environment
from ipad_experiment.data import load_rows,frame_paths
from ipad_experiment.subspace import Moments,Scorer
from subspacead.core.extractor import FeatureExtractor
from subspacead.core.pca import PCAModel
from subspacead.post_process.scoring import calculate_anomaly_scores,post_process_map,aggregate_image_score
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
cfg=read_json(ROOT/"configs/experiment.json");encoder=Encoder(cfg)
row=load_rows("R01",["fit"])[0]
paths=frame_paths(ROOT.parent/"IPAD_dataset",row)
images=[Image.open(paths[i]).convert("RGB") for i in [0,len(paths)//2,len(paths)-1]]
official=FeatureExtractor.__new__(FeatureExtractor);official.model=encoder.model;official.processor=encoder.processor
ref,_,_=official.extract_tokens(images,336,[-4,-5],"mean")
pixels=encoder.processor(images=images,return_tensors="pt",do_resize=True,size={"height":336,"width":336},do_center_crop=False)["pixel_values"].cuda()
with torch.no_grad(): tokens,_,_=encoder.feature_tensors(pixels)
np.testing.assert_allclose(tokens.cpu().numpy().reshape(ref.shape),ref,rtol=1e-6,atol=1e-6)
x=ref.reshape(-1,ref.shape[-1]);pca=PCAModel(ev=.99);params=pca.fit(lambda:iter([x]),x.shape[-1],len(x),1)
stats=Moments(x.shape[-1],"cuda");stats.update(tokens);ours=stats.pca(.99)
assert params['k']==ours['k']
projector_error=float(np.abs(params['components']@params['components'].T-ours['components']@ours['components'].T).max())
assert projector_error<1e-8
patch=calculate_anomaly_scores(x,params,"reconstruction").reshape(3,24,24)
expected=np.array([aggregate_image_score(post_process_map(m,336),"mtop1p") for m in patch])
with torch.no_grad(): actual=Scorer(ours)(tokens)
np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)
result={"status":"passed","environment":environment(),"feature_max_abs_error":float(np.abs(tokens.cpu().numpy().reshape(ref.shape)-ref).max()),"pca_projector_max_abs_error":projector_error,"score_max_abs_error":float(np.abs(actual-expected).max()),"official_scores":expected.tolist(),"our_scores":actual.tolist(),"source_video":row['id'],"frames_tested":[0,len(paths)//2,len(paths)-1]}
write_json(ROOT/"results/00_prepare/gpu_parity.json",result);print(result)
