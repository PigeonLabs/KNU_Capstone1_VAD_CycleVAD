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
for doc in [ROOT/"README.md", *ROOT.glob("results/*/report.md")]:
    for link in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
        if '://' not in link and not link.startswith('#'):
            assert (doc.parent/link.split('#')[0]).exists(),f"{doc}: {link}"
print(f"Artifact policy and report links passed: {len(files)} files")
