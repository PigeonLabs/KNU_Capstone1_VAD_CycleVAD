"""Check the intended tracked artifact set and local README links before push."""
from pathlib import Path
import re,subprocess
ROOT=Path(__file__).resolve().parents[1]
files=subprocess.check_output(["git","ls-files","-co","--exclude-standard"],cwd=ROOT,text=True).splitlines()
for name in files:
    p=ROOT/name
    if not p.is_file():continue
    assert not any(x in p.parts for x in ["artifacts","IPAD_dataset",".venv"]),name
    assert p.suffix not in [".pt",".pth",".safetensors",".joblib",".npy",".npz",".mp4",".jpg"],name
    assert p.stat().st_size<20*1024**2, f"File requires sharding: {name}"
text=(ROOT/"README.md").read_text()
for link in re.findall(r'\]\(([^)]+)\)',text):
    if '://' not in link and not link.startswith('#'):
        assert (ROOT/link.split('#')[0]).exists(),link
print(f"Artifact policy and README links passed: {len(files)} files")
