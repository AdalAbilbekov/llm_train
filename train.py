import pdb
from omegaconf import OmegaConf
from get_model import load_model, save_model
from prepare_data import build_custom_dataset
import fire
import torch
from torch.utils.data import DataLoader
import tqdm
from accelerate import Accelerator
from policies import (
    setup,
    clear_gpu_cache
    )
import os
from time import sleep
from datetime import datetime

def main(config_path):
    conf = OmegaConf.load(config_path)

    if not conf.enable_fsdp:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
    else:
        setup()
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        if torch.distributed.is_initialized(): #<- without this part all shards will be loaded to the single GPU.
            torch.cuda.set_device(local_rank)
            clear_gpu_cache(local_rank)
    if rank == 0:
        print(conf)     
    arguments = {
        "rank": rank if conf.enable_fsdp else None,
        "device": device if not conf.enable_fsdp else None
        }

    # TODO: add logging localy to txt file or so.
    if rank == 0 and conf.log.enable_wandb:
        wandb.init(
            project=f"{conf.log.project_name}",
            name=f"project_name_{datetime.today().strftime('%Y-%m-%d')}",
        )
    
    tokenizer, model, optimizer = load_model(conf, **arguments)

    dataset = build_custom_dataset(conf.dataset_path, tokenizer, conf.max_length, conf.dataset_type)

    dl_train = DataLoader(dataset, shuffle=False, batch_size=conf.batch_size)

    steps_per_epoch = len(dl_train)
    
    # TODO: Add Linear warmup steps 5% of total steps.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=steps_per_epoch, 
        eta_min=conf.min_lr, 
        last_epoch=-1)

    global_step = 0
    epochs = 3
    model.train()
    print("start training")
    for epoch in range(epochs):
        pbar = tqdm.tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=steps_per_epoch, dynamic_ncols=True)
        for step, batch in enumerate(dl_train):
            if not conf.enable_fsdp:
                batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            
            loss = model(**batch).loss
            loss.backward()

            scheduler.step()
            optimizer.step()
    
            pbar.update()

            if rank==0 and conf.log.enable_wandb:
                loss_log = loss.detach().float()
                wandb.log({
                    "train_loss": loss_log,
                    "perplexit":torch.exp(loss_log),
                    "lr": optimizer.param_groups[0]['lr']
                            })

            pbar.set_description(f"Training Epoch: {epoch}, step {step}/{steps_per_epoch} completed (loss: {loss.detach().float()})")

            global_step +=1

            if global_step % conf.save_step == 0:
                save_model(model, optimizer, conf, global_step, rank)
        save_model(model, optimizer, conf, global_step, rank)
        pbar.close()
    
if __name__=="__main__":
    fire.Fire(main("config.yaml"))
    