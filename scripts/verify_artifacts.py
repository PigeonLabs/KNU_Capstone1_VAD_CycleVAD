"""Verify published score coverage, split isolation, and per-stage decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',required=True)
p.add_argument('--data-root',type=Path,help='Also verify each saved GT frame against the frozen source annotation')
args=p.parse_args()
out=ROOT/'results'/args.run
config=json.loads((out/'config.json').read_text())
split=json.loads((ROOT/'results/00_prepare/splits.json').read_text())
if isinstance(split,dict):split=split['videos']
partition='final' if args.run=='03_ablation' else 'development'
seeds=[42] if args.run=='01_baseline' else config['seeds']
expected={(r['scene'],seed,str(r['video'])):r for r in split if r['partition']==partition for seed in seeds}
seen={};hashes={};source_labels={}
for path in sorted((out/'scores').glob('*.csv.gz')):
    d=pd.read_csv(path,dtype={'video':str},float_precision="round_trip")
    key=(d.scene.iloc[0],int(d.seed.iloc[0]),d.video.iloc[0])
    assert key not in seen, f'Duplicate score file {key}'
    row=expected[key]
    assert set(d.partition)=={partition}
    assert len(d)==row['frames']
    np.testing.assert_array_equal(d.frame,np.arange(row['frames']))
    assert set(d['gt']).issubset({0,1})
    assert int(d['gt'].sum())==row['anomaly_frames']
    assert np.isfinite(d.select_dtypes(include='number').to_numpy()).all()
    label_hash=hashlib.sha256(d['gt'].to_numpy(dtype=np.uint8).tobytes()).hexdigest()
    # Original annotation bytes are independently fingerprinted in the preparation
    # manifest; here ensure all seeds retain the identical scalar evaluation labels.
    scene_video=(key[0],key[2])
    if args.data_root is not None:
        if scene_video not in source_labels:
            label_path=args.data_root/row['label_file']
            assert hashlib.sha256(label_path.read_bytes()).hexdigest()==row['label_sha256'], f'Source annotation changed: {scene_video}'
            source_labels[scene_video]=np.load(label_path,allow_pickle=False)
        np.testing.assert_array_equal(d['gt'].to_numpy(),source_labels[scene_video],err_msg=f'Source GT mismatch: {key}')
    if scene_video in hashes:assert hashes[scene_video]==label_hash
    hashes[scene_video]=label_hash
    seen[key]=len(d)
assert set(seen)==set(expected), f'Missing or unexpected evaluation videos: {set(expected)^set(seen)}'
normal={r['id'] for r in split if r['source_split']=='training'}
evaluation={r['id'] for r in expected.values()}
assert not normal&evaluation
if args.run=='02_lora':
    decision=json.loads((out/'backbone_decision.json').read_text())
    passed=decision['mean_delta']>=config['lora_adoption']['macro_auroc_min_delta'] and decision['ci_low']>0
    assert passed==decision['adopt_lora']
    for scene in config['scenes']:
        for seed in seeds:
            metadata=json.loads((out/'training'/f'{scene}_seed{seed}.json').read_text())
            assert metadata['status']=='completed'
            assert metadata['frozen_hash_before']==metadata['frozen_hash_after']
            assert metadata['selection_uses_development_labels'] is False
if args.run=='03_ablation':
    fixed=json.loads((out/'frozen_protocol.json').read_text())
    actual=hashlib.sha256((ROOT/'results/02_lora/backbone_decision.json').read_bytes()).hexdigest()
    assert fixed['backbone_decision_sha256']==actual
    for path in (out/'fit_reports').glob('*.json'):
        fit=json.loads(path.read_text());assert fit['conditional']['rank']==fit['constant']['rank']
result={'status':'passed','partition':partition,'scene_seed_video_files':len(seen),'score_rows_including_seeds':sum(seen.values()),'unique_videos':len(hashes),'unique_frames':sum(r['frames'] for r in split if r['partition']==partition),'seeds':seeds,'checks':['exact expected video coverage','contiguous full stride-1 frame coverage','finite scalar data','identical labels across seeds','normal/evaluation split separation','stage-specific adoption or frozen-protocol checks']}
result['source_annotation_videos_verified']=len(source_labels)
if args.data_root is not None:result['checks'].append('source annotation SHA256 and exact per-frame GT equality')
(out/'artifact_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
