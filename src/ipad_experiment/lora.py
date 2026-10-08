"""Normal-only, fixed-teacher feature denoising; no development-label training."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .encoder import Encoder, Frames, install_lora, adapter_state
from .data import load_rows, frame_paths
from .io import ROOT, write_json, read_json, init_run, log, file_hash, digest

def frozen_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def features(encoder,pixels):
    baseline,cycle,cls=encoder.feature_tensors(pixels)
    return [baseline,*cycle,cls]

def distill_loss(pred,target):
    return torch.stack([((x-y)**2).mean()/(y.square().mean()+1e-8) for x,y in zip(pred,target)]).mean()

def augment(pixels,processor,cfg,generator):
    mean=torch.tensor(processor.image_mean,device=pixels.device)[None,:,None,None]
    std=torch.tensor(processor.image_std,device=pixels.device)[None,:,None,None]
    image=pixels*std+mean;n=len(image)
    def uniform(bounds): return bounds[0]+torch.rand((n,1,1,1),generator=generator,device=pixels.device)*(bounds[1]-bounds[0])
    image=image*uniform(cfg["brightness"])
    gray=(image*image.new_tensor([.2989,.587,.114])[None,:,None,None]).sum(1,keepdim=True).mean((2,3),keepdim=True)
    image=gray+(image-gray)*uniform(cfg["contrast"])
    image=image+torch.randn(image.shape,device=image.device,generator=generator)*uniform(cfg["noise_std"])
    return (image.clamp(0,1)-mean)/std

def sampled_paths(data_root,rows,count,seed,fixed=False):
    rng=np.random.default_rng(seed);result=[]
    for row in rows:
        paths=frame_paths(data_root,row)
        edges=np.linspace(0,len(paths),count+1)
        for a,b in zip(edges[:-1],edges[1:]):
            i=int((a+b)/2) if fixed else int(rng.uniform(a,b))
            result.append(paths[min(i,len(paths)-1)])
    return result

def train_one(data_root,cfg,out,scene,seed):
    c=cfg["lora"];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    folder=ROOT/"artifacts/lora"/scene/f"seed{seed}";folder.mkdir(parents=True,exist_ok=True)
    meta=out/"training"/f"{scene}_seed{seed}.json"
    path=folder/"best.pt"
    identity=digest({"config":cfg,"fit":load_rows(scene,["fit"]),"validation":load_rows(scene,["validation"]),"seed":seed,"algorithm":"teacher_normal_feature_denoising_v1"})
    if meta.exists() and path.exists():
        previous=read_json(meta)
        if previous.get("status")=="completed" and previous["identity"]==identity and previous["checkpoint_sha256"]==file_hash(path):
            log(out/"logs"/"training.jsonl",event="training_cache_hit",scene=scene,seed=seed)
            return path
    student=Encoder(cfg);teacher=Encoder(cfg)
    targets=install_lora(student.model,c)
    # New adapters were created on CPU after the model move.
    student.model.to("cuda").eval();teacher.model.eval()
    weights=[p for p in student.model.parameters() if p.requires_grad]
    initial_hash=frozen_hash(student.model)
    optimizer=torch.optim.AdamW(weights,lr=c["learning_rate"],weight_decay=c["weight_decay"])
    val_paths=sampled_paths(data_root,load_rows(scene,["validation"]),c["validation_frames_per_video"],seed,fixed=True)
    val_loader=DataLoader(Frames(val_paths,student.processor,cfg["encoder"]["image_size"]),batch_size=c["micro_batch"],num_workers=cfg["encoder"]["workers"],shuffle=False,pin_memory=True)
    def validate():
        rng=torch.Generator(device="cuda").manual_seed(424242)
        total=0.;seen=0
        with torch.no_grad():
            for clean in val_loader:
                clean=clean.cuda(non_blocking=True);target=features(teacher,clean)
                loss=distill_loss(features(student,augment(clean,student.processor,c,rng)),target)+c["clean_weight"]*distill_loss(features(student,clean),target)
                total+=loss.item()*len(clean);seen+=len(clean)
        return total/seen
    state_path=folder/"resume.pt";first=0;stale=0;best=math.inf;best_epoch=-1;step=0;elapsed_before=0.
    training_rows=load_rows(scene,["fit"])
    n_samples=len(training_rows)*c["frames_per_video"]
    steps_per_epoch=math.ceil(n_samples/c["effective_batch"])
    if state_path.exists():
        state=torch.load(state_path,map_location="cpu",weights_only=False)
        if state["identity"]!=identity:raise ValueError("Training resume identity differs")
        student.model.load_state_dict(state["adapter"],strict=False);optimizer.load_state_dict(state["optimizer"])
        first=state["next_epoch"];best=state["best"];best_epoch=state["best_epoch"];stale=state["stale"];step=state["step"];elapsed_before=state["elapsed_seconds"]
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    if first==0:
        initial_val=validate();log(out/"logs"/f"{scene}_seed{seed}_epochs.jsonl",event="initial_validation",epoch=-1,validation_loss=initial_val)
    epoch=first-1
    remaining=range(first,c["epochs"]) if stale<c["patience"] else range(0)
    for epoch in remaining:
        torch.manual_seed(seed+epoch*1009)
        paths=sampled_paths(data_root,training_rows,c["frames_per_video"],seed+epoch*1009)
        generator=torch.Generator().manual_seed(seed+epoch*1009)
        loader=DataLoader(Frames(paths,student.processor,cfg["encoder"]["image_size"]),batch_size=c["micro_batch"],num_workers=cfg["encoder"]["workers"],shuffle=True,pin_memory=True,generator=generator)
        rng=torch.Generator(device="cuda").manual_seed(seed+epoch*1009)
        total=0.;seen=0;group_seen=0;optimizer.zero_grad(set_to_none=True)
        epoch_start=time.perf_counter()
        for clean in loader:
            clean=clean.cuda(non_blocking=True);b=len(clean)
            group_target=min(c["effective_batch"],n_samples-(seen-group_seen))
            with torch.no_grad():target=features(teacher,clean)
            loss_aug=distill_loss(features(student,augment(clean,student.processor,c,rng)),target)
            loss_clean=distill_loss(features(student,clean),target)
            loss=loss_aug+c["clean_weight"]*loss_clean
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite LoRA training loss")
            (loss*b/group_target).backward();total+=float(loss.detach())*b;seen+=b;group_seen+=b
            if group_seen==group_target:
                warm=steps_per_epoch*c["warmup_epochs"];total_steps=steps_per_epoch*c["epochs"]
                factor=(step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,total_steps-warm)))
                for group in optimizer.param_groups:group["lr"]=c["learning_rate"]*factor
                optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1;group_seen=0
        val=validate();torch.cuda.synchronize();duration=time.perf_counter()-epoch_start
        if val<best:
            best=val;best_epoch=epoch;stale=0
            temporary=folder/"best.tmp.pt";torch.save(adapter_state(student.model),temporary);temporary.replace(path)
        else:stale+=1
        elapsed=elapsed_before+time.perf_counter()-start
        log(out/"logs"/f"{scene}_seed{seed}_epochs.jsonl",event="epoch",scene=scene,seed=seed,epoch=epoch,train_loss=total/seen,validation_loss=val,seconds=duration,best_epoch=best_epoch,learning_rate=optimizer.param_groups[0]["lr"],peak_vram_bytes=torch.cuda.max_memory_allocated())
        resume={"identity":identity,"next_epoch":epoch+1,"adapter":adapter_state(student.model),"optimizer":optimizer.state_dict(),"best":best,"best_epoch":best_epoch,"stale":stale,"step":step,"elapsed_seconds":elapsed}
        temp=folder/"resume.tmp.pt";torch.save(resume,temp);temp.replace(state_path)
        if stale>=c["patience"]:break
    final_hash=frozen_hash(student.model)
    if final_hash!=initial_hash:raise RuntimeError("Frozen backbone changed during training")
    record={"status":"completed","identity":identity,"scene":scene,"seed":seed,"best_epoch":best_epoch,"best_normal_validation_loss":best,"epochs_completed":epoch+1,
            "checkpoint_sha256":file_hash(path),"trainable_parameters":sum(p.numel() for p in weights),"lora_targets":targets,"frozen_hash_before":initial_hash,"frozen_hash_after":final_hash,
            "seconds":elapsed_before+time.perf_counter()-start,"peak_vram_bytes":int(torch.cuda.max_memory_allocated()),"fit_videos":len(training_rows),"validation_videos":len(load_rows(scene,["validation"])),"selection_uses_development_labels":False}
    write_json(meta,record)
    del student,teacher,optimizer;torch.cuda.empty_cache();return path

def run_training(data_root,cfg,scenes=None):
    out=init_run("02_lora",cfg)
    # Global adoption and completion are performed by select-backbone, after all 12 fits.
    for scene in scenes or cfg["scenes"]:
        for seed in cfg["seeds"]:train_one(data_root,cfg,out,scene,seed)
    write_json(out/"training_status.json",{"status":"completed" if scenes is None else "partial","scenes":scenes or cfg["scenes"],"seeds":cfg["seeds"]})
