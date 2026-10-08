"""Execute the portable analysis notebook with the current Python environment."""
import os,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
prefix=ROOT/"artifacts/jupyter"
os.environ["JUPYTER_PATH"]=str(prefix/"share/jupyter")
os.environ["IPYTHONDIR"]=str(ROOT/"artifacts/ipython")
os.environ["MPLCONFIGDIR"]=str(ROOT/"artifacts/matplotlib")
subprocess.run([sys.executable,"-m","ipykernel","install","--prefix",str(prefix),"--name","ipad-experiment","--display-name","IPAD Experiment"],check=True)
import nbformat
from nbclient import NotebookClient
p=ROOT/"notebooks/Experiment.ipynb"
n=nbformat.read(p,as_version=4)
NotebookClient(n,timeout=600,kernel_name="ipad-experiment",resources={"metadata":{"path":str(ROOT)}}).execute()
nbformat.validate(n);nbformat.write(n,p)
print("Executed and saved:",p.name)
