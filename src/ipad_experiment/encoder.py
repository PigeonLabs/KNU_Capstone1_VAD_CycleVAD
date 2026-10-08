"""One differentiable DINOv2 forward for the official and cycle descriptors."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, Dinov2Model
from .io import ROOT, digest
from .data import frame_paths

REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"

class Frames(Dataset):
    def __init__(self, paths, processor, size):
        self.paths, self.processor, self.size = paths, processor, size
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im: image = im.convert("RGB")
        return self.processor(images=image, return_tensors="pt", do_resize=True,
                              size={"height": self.size, "width": self.size}, do_center_crop=False)["pixel_values"][0]

class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__(); self.base = base; self.scale=alpha/rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=np.sqrt(5))
        self.base.requires_grad_(False)
    def forward(self, x): return self.base(x) + F.linear(F.linear(x, self.a), self.b)*self.scale

def install_lora(model, cfg):
    model.requires_grad_(False)
    installed=[]
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.split(".")[-1] in cfg["targets"]:
            parent, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), leaf, LoRALinear(module, cfg["rank"], cfg["alpha"]))
            installed.append(name)
    if len(installed) != 2*model.config.num_hidden_layers: raise RuntimeError("Unexpected LoRA target topology")
    return installed

def adapter_state(model):
    return {k: v.detach().cpu() for k,v in model.state_dict().items() if k.endswith((".a", ".b"))}

class Encoder:
    def __init__(self, cfg, adapter=None, device="cuda"):
        self.cfg=cfg["encoder"]; self.device=torch.device(device)
        if device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable: use the verified host execution context; no silent CPU fallback")
        self.processor=AutoImageProcessor.from_pretrained(self.cfg["model_id"], revision=REVISION, cache_dir=ROOT/"artifacts/hf", local_files_only=True, use_fast=False)
        self.model=Dinov2Model.from_pretrained(self.cfg["model_id"], revision=REVISION, cache_dir=ROOT/"artifacts/hf", local_files_only=True, attn_implementation="eager").eval()
        self.model.requires_grad_(False)
        self.adapter_hash=None
        if adapter:
            from .io import file_hash
            self.adapter_hash=file_hash(adapter)
            install_lora(self.model,cfg["lora"])
            state=torch.load(adapter,map_location="cpu",weights_only=True)
            expected=set(adapter_state(self.model)); actual=set(state)
            if expected != actual: raise ValueError("Adapter keys do not match model")
            self.model.load_state_dict(state,strict=False)
            self.model.requires_grad_(False)
        self.model.to(self.device)
        channels=self.model.config.hidden_size*len(self.cfg["cycle_layers"])
        rng=np.random.default_rng(self.cfg["projection_seed"])
        self.projection=torch.tensor(rng.standard_normal((channels,self.cfg["patch_dim"]))/np.sqrt(self.cfg["patch_dim"]),dtype=torch.float32,device=self.device)
        self.identity={**self.cfg,"revision":REVISION,"adapter_sha256":self.adapter_hash,"implementation":1}

    def feature_tensors(self, pixels):
        out=self.model(pixel_values=pixels,output_hidden_states=True,return_dict=True)
        baseline=torch.stack([out.hidden_states[l][:,1:] for l in self.cfg["baseline_layers"]]).mean(0)
        cycle=[out.hidden_states[l][:,1:] for l in self.cfg["cycle_layers"]]
        cls=out.last_hidden_state[:,0]
        return baseline,cycle,cls

    def descriptors(self, cycle, cls):
        side=self.cfg["image_size"]//self.model.config.patch_size; grid=self.cfg["patch_grid"]
        layers=[]
        for t in cycle:
            spatial=F.normalize(t.float(),dim=-1).transpose(1,2).reshape(len(t),-1,side,side)
            spatial=F.adaptive_avg_pool2d(spatial,(grid,grid)).flatten(2).transpose(1,2)
            layers.append(F.normalize(spatial,dim=-1))
        local=F.normalize(torch.cat(layers,dim=-1)@self.projection,dim=-1)
        return F.normalize(cls.float(),dim=-1),local

    def batches(self, data_root, row, batch_size=None):
        dataset=Frames(frame_paths(data_root,row),self.processor,self.cfg["image_size"])
        loader=DataLoader(dataset,batch_size=batch_size or self.cfg["batch_size"],num_workers=self.cfg["workers"],shuffle=False,pin_memory=self.device.type=="cuda")
        with torch.inference_mode():
            for pixels in loader:
                tokens,cycle,cls=self.feature_tensors(pixels.to(self.device,non_blocking=True))
                global_z,local=self.descriptors(cycle,cls)
                yield tokens,global_z.cpu().numpy(),local.cpu().numpy()

    def signature(self,row): return digest({"encoder":self.identity,"id":row["id"],"frames":row["frames"],"frame_listing_sha256":row["frame_listing_sha256"]})
