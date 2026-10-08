"""Independent sklearn/event verification from published scalar CSVs (no GPU)."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
out=ROOT/'results'/args.run
reported=pd.read_csv(out/'metrics_by_scene.csv',float_precision="round_trip");tables=[]
for p in sorted((out/'scores').glob('*.csv.gz')):tables.append(pd.read_csv(p,dtype={'video':str},float_precision="round_trip"))
all_scores=pd.concat(tables,ignore_index=True);checks=[]
for _,r in reported.iterrows():
    d=all_scores[(all_scores.scene==r.scene)&(all_scores.seed==r.seed)&(all_scores.partition==r.partition)].sort_values(['video','frame'])
    assert not d.duplicated(['video','frame']).any()
    y=d['gt'].to_numpy();s=d[r.branch].to_numpy();prediction=s>=r.threshold
    values={'auroc':roc_auc_score(y,s),'ap':average_precision_score(y,s),'f1':f1_score(y,prediction,zero_division=0),'tpr':prediction[y==1].mean(),'fpr':prediction[y==0].mean()}
    events=detected=delay_total=0
    for _,video in d.groupby('video'):
        label=video['gt'].to_numpy();hit=video[r.branch].to_numpy()>=r.threshold
        previous=0;start=None
        for i,value in enumerate(np.r_[label,0]):
            if value==1 and previous==0:start=i
            if value==0 and previous==1:
                events+=1;matches=np.flatnonzero(hit[start:i])
                if len(matches):detected+=1;delay_total+=int(matches[0])
            previous=value
    values.update(events=events,detected_events=detected,delay_sum=delay_total)
    for metric,value in values.items():np.testing.assert_allclose(value,r[metric],rtol=1e-9,atol=1e-10,err_msg=f'{r.scene}/{r.seed}/{r.branch}/{metric}')
    checks.append({'scene':r.scene,'seed':int(r.seed),'branch':r.branch,'partition':r.partition,'frames':len(d),'status':'passed'})
result={'status':'passed','method':'independent sklearn metrics and per-video event scan from published CSV','checks':checks}
(out/'metric_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(f'Verified {len(checks)} scene/seed/branch rows from {len(all_scores)} score rows')
