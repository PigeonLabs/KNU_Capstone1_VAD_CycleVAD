"""Streaming sufficient statistics and the official reconstruction score path."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from .io import ROOT
sys.path.insert(0,str(ROOT/"vendor"))
from subspacead.post_process.scoring import post_process_map, aggregate_image_score

class Moments:
    def __init__(self, dim, device):
        self.n=0; self.mean=torch.zeros(dim,dtype=torch.float64,device=device)
        self.m2=torch.zeros((dim,dim),dtype=torch.float64,device=device)
    def update(self,x):
        x=x.reshape(-1,x.shape[-1]).double(); n=len(x); mean=x.mean(0); delta=mean-self.mean
        centered=x-mean; self.m2+=centered.T@centered+torch.outer(delta,delta)*(self.n*n/max(1,self.n+n))
        self.mean+=delta*(n/max(1,self.n+n)); self.n+=n
    def merge(self,n,mean,m2):
        mean=torch.as_tensor(mean,dtype=torch.float64,device=self.mean.device)
        m2=torch.as_tensor(m2,dtype=torch.float64,device=self.mean.device)
        delta=mean-self.mean; self.m2+=m2+torch.outer(delta,delta)*(self.n*n/(self.n+n))
        self.mean+=delta*(n/(self.n+n)); self.n+=n
    def arrays(self): return {"n":np.array(self.n),"mean":self.mean.cpu().numpy(),"m2":self.m2.cpu().numpy()}
    def pca(self,variance=.99):
        eigen,basis=torch.linalg.eigh(self.m2/(self.n-1)); eigen=eigen.flip(0); basis=basis.flip(1)
        fraction=torch.cumsum(eigen,0)/eigen.sum()
        k=int(torch.searchsorted(fraction,torch.tensor(variance,dtype=eigen.dtype,device=eigen.device)).item())+1
        return {"mu":self.mean.cpu().numpy(),"components":basis[:,:k].cpu().numpy(),"eigvals":eigen[:k].cpu().numpy(),
                "k":k,"n":self.n,"retained_variance":float(fraction[k-1]),"eps":1e-6,"whiten":False}

class Scorer:
    def __init__(self,params,device="cuda",image_size=336):
        self.params=params;self.size=image_size
        # The official scorer casts PCA arrays to the feature dtype.
        self.mean=torch.as_tensor(params["mu"],dtype=torch.float32,device=device)
        self.basis=torch.as_tensor(params["components"],dtype=torch.float32,device=device)
    def __call__(self,tokens):
        centered=tokens-self.mean
        reconstructed=(centered@self.basis)@self.basis.T+self.mean
        patch=((tokens-reconstructed)**2).sum(-1)
        side=int(np.sqrt(patch.shape[1])); maps=patch.reshape(-1,side,side).cpu().numpy()
        return np.array([aggregate_image_score(post_process_map(m,self.size),"mtop1p") for m in maps])
