"""A real normal-frame training-step smoke check, not an IPAD performance result."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ipad_experiment.io import ROOT,read_json,write_json
from ipad_experiment.encoder import Encoder,Frames,install_lora,adapter_state
from ipad_experiment.data import load_rows,frame_paths
from ipad_experiment.lora import features,augment,distill_loss,frozen_hash
torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
cfg=read_json(ROOT/'configs/experiment.json');student=Encoder(cfg);teacher=Encoder(cfg)
targets=install_lora(student.model,cfg['lora']);student.model.cuda()
before=frozen_hash(student.model)
paths=frame_paths(ROOT.parent/'IPAD_dataset',load_rows('R01',['fit'])[0])[:4]
clean=next(iter(DataLoader(Frames(paths,student.processor,336),batch_size=4))).cuda()
with torch.no_grad():target=features(teacher,clean)
torch.testing.assert_close(features(student,clean)[0],target[0],rtol=0,atol=0)
params=[p for p in student.model.parameters() if p.requires_grad]
optimizer=torch.optim.AdamW(params,lr=1e-4)
rng=torch.Generator(device='cuda').manual_seed(42)
loss=distill_loss(features(student,augment(clean,student.processor,cfg['lora'],rng)),target)+.1*distill_loss(features(student,clean),target)
loss.backward();grad_norm=float(torch.stack([p.grad.square().sum() for p in params if p.grad is not None]).sum().sqrt())
assert grad_norm>0 and torch.isfinite(loss)
optimizer.step();after=frozen_hash(student.model);assert before==after
assert any(v.abs().sum()>0 for k,v in adapter_state(student.model).items() if k.endswith('.b'))
write_json(ROOT/'results/00_prepare/lora_gpu_smoke.json',{'status':'passed','loss':float(loss.detach()),'gradient_norm':grad_norm,'frozen_unchanged':before==after,'trainable_parameters':sum(p.numel() for p in params),'targets':targets,'frames':len(paths),'performance_experiment':False})
print('LoRA real-frame gradient smoke passed:',float(loss.detach()),grad_norm)
