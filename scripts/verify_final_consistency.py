"""Cross-check final contrasts, seed coverage, and the unchanged frozen baseline."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ipad_experiment.io import ROOT, digest, file_hash, read_json, write_json

out = ROOT / 'results/03_ablation'
cfg = read_json(out / 'config.json')
protocol = read_json(out / 'frozen_protocol.json')
assert protocol['config_sha256'] == digest(cfg)
assert protocol['backbone_decision_sha256'] == file_hash(ROOT / 'results/02_lora/backbone_decision.json')
assert read_json(ROOT / 'results/02_lora/backbone_decision.json')['selected_backbone'] == 'frozen'
read = lambda p: pd.read_csv(p, dtype={'video': str}, float_precision='round_trip')
metrics = read(out / 'metrics_by_scene.csv')
expected = {(scene, seed, branch) for scene in cfg['scenes'] for seed in cfg['seeds'] for branch in protocol['branches']}
assert set(zip(metrics.scene, metrics.seed, metrics.branch)) == expected
assert len(metrics) == len(expected)
fits = {p.stem: read_json(p) for p in (out / 'fit_reports').glob('*.json')}
assert set(fits) == {f'{s}_seed{k}' for s in cfg['scenes'] for k in cfg['seeds']}
for fit in fits.values():
    best = min(fit['selection'], key=lambda r: r['normal_validation_mse'])
    assert (fit['harmonics'], fit['ridge']) == (best['harmonics'], best['ridge'])
    assert fit['conditional']['rank'] == fit['constant']['rank']
    assert fit['conditional']['samples'] == fit['constant']['samples']
    assert fit['backbone'] == 'frozen'

frozen = {}
for path in (out / 'frozen_baseline/scores').glob('*.csv.gz'):
    d = read(path)
    key = (d.scene.iloc[0], d.video.iloc[0])
    assert key not in frozen
    frozen[key] = d
seen = set()
for path in (out / 'scores').glob('*.csv.gz'):
    d = read(path)
    key = (d.scene.iloc[0], d.video.iloc[0])
    seed = int(d.seed.iloc[0])
    np.testing.assert_array_equal(d[['frame', 'gt', 'raw_S', 'S']].to_numpy(), frozen[key][['frame', 'gt', 'raw_S', 'S']].to_numpy())
    seen.add((*key, seed))
    baseline_fit = read_json(out / 'frozen_baseline/fit_reports' / f'{key[0]}_seed42.json')
    assert fits[f'{key[0]}_seed{seed}']['thresholds']['S'] == baseline_fit['thresholds']['S']
assert seen == {(*key, seed) for key in frozen for seed in cfg['seeds']}
assert len(frozen) == 37

macro = metrics.groupby('branch').auroc.mean()
comparisons = read(out / 'comparisons.csv').set_index('contrast')
names = [f'{b}_minus_{a}' for a, b in protocol['contrasts']]
assert set(comparisons.index) == set(names) and len(comparisons) == len(names)
for a, b in protocol['contrasts']:
    name = f'{b}_minus_{a}'
    result = read_json(out / 'comparisons' / f'{name}.json')
    np.testing.assert_allclose(comparisons.loc[name, 'mean_delta'], macro[b] - macro[a], atol=1e-14, rtol=0)
    np.testing.assert_allclose(comparisons.loc[name, ['mean_delta', 'ci_low', 'ci_high']].to_numpy(dtype=float), [result[k] for k in ['mean_delta', 'ci_low', 'ci_high']], atol=1e-14, rtol=0)
    assert result['bootstrap_samples'] == cfg['bootstrap']['samples']
    assert result['bootstrap_seed'] == cfg['bootstrap']['seed']

result = {'status': 'passed', 'scene_seed_branch_rows': len(metrics), 'frozen_videos': len(frozen), 'matched_scene_seed_video_files': len(seen), 'prespecified_contrasts': len(names), 'checks': ['frozen decision and configuration identity', 'complete scene/seed/branch and model coverage', 'minimum normal-validation mean selection', 'matched C/C0 rank and sample count', 'exact frozen baseline frame/GT/raw/calibrated score equality across seeds', 'unchanged baseline thresholds', 'contrast deltas recovered independently from metrics and JSON/CSV agreement']}
write_json(out / 'consistency_verification.json', result)
print(json.dumps(result))
