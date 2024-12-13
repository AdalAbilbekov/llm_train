from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import pdb
from omegaconf import OmegaConf

import torch
import torch.optim as optim
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from policies import (
    get_policies, 
    apply_fsdp_checkpointing,
    setup,
    clear_gpu_cache
    )

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp import ShardingStrategy
import torch.distributed as dist
import os

from time import sleep

def set_model(model, model_config, rank):
    if model_config.enable_fsdp:
        mixed_precision_policy, wrapping_policy, to_use_bfloat = get_policies(model_config, rank)

        # Use pure BF16
        if model_config.brain_float and to_use_bfloat: model.to(torch.bfloat16)
        
        model = FSDP(
            model,
            auto_wrap_policy=wrapping_policy,
            mixed_precision=mixed_precision_policy if not model_config.brain_float else None,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
        )

        apply_fsdp_checkpointing(model)
        
        if rank == 0:
            print(model)

        return model
    else:
        return model

def load(model_config: None):
    use_cache = False if model_config.enable_fsdp else True
    # pdb.set_trace()
    assert model_config.model.path != None, "There is no models provided."

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model.path,
        load_in_8bit=True if model_config.quantization else False,
        device_map="auto" if model_config.quantization else None,
        use_cache=use_cache,
    )

    return model

# TODO: Prepare optimizer for the FSDP train.
def load_optimizer(model, model_config):
    return optim.AdamW(
        model.parameters(),
        lr=model_config.lr,
        # weight_decay=model_config.weight_decay
    )

def load_model(model_config, rank):
    model = load(model_config)
    model = set_model(model, model_config, rank)

    tokenizer = AutoTokenizer.from_pretrained(model_config.model.path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # optimizer = load_optimizer(model, model_config)
    print("let's go to the next era")
    if rank==0:
        pdb.set_trace()
    sleep(3000)
    return tokenizer, model, optimizer
    
if __name__=="__main__":
    conf = OmegaConf.load("config.yaml")
    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    if torch.distributed.is_initialized(): #<- without this part all shards will be loaded to the single GPU.
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
    tokenizer, model, optmizer = load_model(conf, rank)