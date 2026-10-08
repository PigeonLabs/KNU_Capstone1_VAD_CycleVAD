"""Audit completed LoRA fits against epoch logs and local adapter checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');p.add_argument('--checkpoints',action='store_true');args=p.parse_args()
out=ROOT/'results/02_lora';config=json.loads((out/'config.json').read_text());checks=[];missing=[]
for scene in config['scenes']:
 for seed in config['seeds']:
  path=out/'training'/f'{scene}_seed{seed}.json'
  if not path.exists():missing.append(f'{scene}/seed{seed}');continue
  record=json.loads(path.read_text())
  assert record['status']=='completed'
  assert record['frozen_hash_before']==record['frozen_hash_after']
  assert record['selection_uses_development_labels'] is False
  lines=[json.loads(s) for s in (out/'logs'/f'{scene}_seed{seed}_epochs.jsonl').read_text().splitlines()]
  epochs=[s for s in lines if s['event']=='epoch']
  assert [s['epoch'] for s in epochs]==list(range(record['epochs_completed']))
  assert 1<=len(epochs)<=config['lora']['epochs']
  best=min(epochs,key=lambda s:s['validation_loss'])
  assert best['epoch']==record['best_epoch']
  assert best['validation_loss']==record['best_normal_validation_loss']
  if len(epochs)<config['lora']['epochs']:
   assert len(epochs)-1-best['epoch']>=config['lora']['patience']
  if args.checkpoints:
   checkpoint=ROOT/'artifacts/lora'/scene/f'seed{seed}/best.pt'
   assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==record['checkpoint_sha256']
   tensors=torch.load(checkpoint,map_location='cpu',weights_only=True)
   expected={f'{target}.{suffix}' for target in record['lora_targets'] for suffix in ['a','b']}
   assert set(tensors)==expected
   assert sum(t.numel() for t in tensors.values())==record['trainable_parameters']
   assert all(torch.isfinite(t).all() for t in tensors.values())
   assert any(torch.count_nonzero(t)>0 for name,t in tensors.items() if name.endswith('.b'))
  checks.append({'scene':scene,'seed':seed,'epochs':len(epochs),'best_epoch':best['epoch'],'status':'passed'})
if not args.partial:assert not missing,f'Missing completed fits: {missing}'
result={'status':'partial' if missing else 'passed','completed_fits':len(checks),'expected_fits':len(config['scenes'])*len(config['seeds']),'local_checkpoints_verified':args.checkpoints and bool(checks),'checkpoint_fits_verified':len(checks) if args.checkpoints else 0,'missing':missing,'checks':checks,'criteria':['contiguous epoch history','maximum epochs / early stopping respected','checkpoint selected by minimum normal validation loss','frozen parameter fingerprints unchanged','adapter checkpoint hash, parameter names, finite values and nonzero learned update when requested']}
(out/'training_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
