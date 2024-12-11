import pdb
from omegaconf import OmegaConf
from get_model import load_model
from prepare_data import build_custom_dataset
import fire
import torch
from torch.utils.data import DataLoader
import tqdm
from accelerate import Accelerator

def main(config_path):
    conf = OmegaConf.load(config_path)

    tokenizer, model, optimizer = load_model(conf)

    dataset = build_custom_dataset(conf.dataset_path, tokenizer, conf.max_length, conf.dataset_type)

    dl_train = DataLoader(dataset, shuffle=False, batch_size=2)
    
    if conf.enable_fsdp:
        accelerator = Accelerator()
        dl_train, model, optimizer = accelerator.prepare(
            dl_train, 
            model, 
            optimizer)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    global_step = 0
    epochs = 3
    steps_per_epoch = len(dl_train)
    model.train()
    print("start training")
    for epoch in range(epochs):
        pbar = tqdm.tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=steps_per_epoch, dynamic_ncols=True)
        for step, batch in enumerate(dl_train):
            if not conf.enable_fsdp:
                batch = {k: v.to(device) for k, v in batch.items()}
            
            loss = model(**batch).loss

            if conf.enable_fsdp:
                accelerator.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.update()
            pbar.set_description(f"Training Epoch: {epoch}, step {step}/{steps_per_epoch} completed (loss: {loss.detach().float()})")

            global_step +=1

            if global_step % conf.save_step == 0:
                print(f"Saving a checkpont {conf.save_path}_{global_step}")
                model.save_pretrained(f"{conf.save_path}_{global_step}", from_pt=True)
        pbar.close()
    
if __name__=="__main__":
    fire.Fire(main("config.yaml"))
    