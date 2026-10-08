"""Wait for an existing Linux training process, then verify/evaluate stage 02.

This does not restart training, publish to GitHub, or start final ablations.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from ipad_experiment.io import ROOT,write_json,stamp

p=argparse.ArgumentParser()
p.add_argument('--training-pid',type=int,required=True)
p.add_argument('--data-root',type=Path,required=True)
a=p.parse_args()
proc=Path('/proc')/str(a.training_pid)

def process_state():
    try:
        fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        return fields[0],fields[19]
    except FileNotFoundError:return None

initial=process_state()
if initial is None:raise RuntimeError('Training PID is not live; inspect its outcome before running selection directly')
cmdline=(proc/'cmdline').read_bytes().split(b'\0')
if b'ipad_experiment.pipeline' not in cmdline or b'train-lora' not in cmdline:
    raise RuntimeError('PID is not the expected training command')
out=ROOT/'results/02_lora'
write_json(out/'handoff_status.json',{'status':'waiting_for_training','training_pid':a.training_pid,'process_start_ticks':initial[1],'started_utc':stamp()})
print(f'Waiting for existing training PID {a.training_pid}; no training restart.',flush=True)
while True:
    state=process_state()
    if state is None or state[1]!=initial[1] or state[0]=='Z':break
    time.sleep(5)
status=out/'training_status.json'
if not status.exists() or json.loads(status.read_text()).get('status')!='completed':
    write_json(out/'handoff_status.json',{'status':'training_incomplete','observed_utc':stamp()})
    raise RuntimeError('Training ended without all requested fits; evaluation was not started')
steps=[
 ['scripts/verify_training.py','--checkpoints'],
 ['-m','ipad_experiment.pipeline','select-backbone','--data-root',str(a.data_root)],
 ['scripts/recompute_metrics.py','--run','02_lora'],
 ['scripts/verify_artifacts.py','--run','02_lora'],
 ['-m','ipad_experiment.pipeline','report','--run','02_lora'],
 ['scripts/execute_notebook.py'],
 ['scripts/check_publish.py'],
]
for step in steps:
    write_json(out/'handoff_status.json',{'status':'running_postprocess','command':step,'started_utc':stamp()})
    try:subprocess.run([sys.executable,*step],cwd=ROOT,check=True)
    except subprocess.CalledProcessError as error:
        write_json(out/'handoff_status.json',{'status':'failed','command':step,'exit_code':error.returncode,'observed_utc':stamp()})
        raise
write_json(out/'handoff_status.json',{'status':'ready_for_review_and_publication','completed_utc':stamp(),'final_ablation_started':False})
print('Stage 02 is ready for figure review and GitHub publication. Final ablation remains gated.',flush=True)
