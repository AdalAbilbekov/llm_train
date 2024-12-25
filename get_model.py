from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import pdb
from omegaconf import OmegaConf

import torch
import torch.optim as optim
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictConfig,
    StateDictType
)
from policies import (
    get_policies, 
    apply_fsdp_checkpointing,
    setup,
    clear_gpu_cache
    )

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
import torch.distributed as dist
import os

from time import sleep

def set_model(model, model_config, rank, device=None):
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
            forward_prefetch=True
        )

        apply_fsdp_checkpointing(model)
        
        if rank == 0:
            print(model)

        return model
    else:
        return model.to(device)

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

# Prepare optimizer for the FSDP train.
def load_optimizer(model, model_config):
    return optim.AdamW(
        model.parameters(),
        lr=model_config.lr,
        weight_decay=model_config.weight_decay
    )

# TODO: save optimizer, and save checkpoints in format of .safetensors.
def save_model(model, optimizer, model_config, step, rank):
    if model_config.enable_fsdp:
        fullstate_save_policy = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
            )

        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
        ):
            cpu_state=model.state_dict()
            print(f"saving process: rank {rank} with model state_dict\n")

        if rank==0:
            torch.save(cpu_state, f"{model_config.model.save_path}_{step}.pt")
            print(f"model checkpoint saved at {model_config.model.save_path}_{step}\n")
    else:
        model.save_pretrained(f"{model_config.model.save_path}_{step}.pt")
        print(f"model checkpoint saved at {model_config.model.save_path}_{step}\n")

def load_model(model_config, **kwargs):
    rank = kwargs.get("rank", None)
    device = kwargs.get("device", None)

    model = load(model_config)
    model = set_model(model, model_config, rank, device)

    tokenizer = AutoTokenizer.from_pretrained(model_config.model.path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    optimizer = load_optimizer(model, model_config)
    
    return tokenizer, model, optimizer
    
if __name__=="__main__":
    conf = OmegaConf.load("config.yaml")
    if not conf.enable_fsdp:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        setup()
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        if torch.distributed.is_initialized(): #<- without this part all shards will be loaded to the single GPU.
            torch.cuda.set_device(local_rank)
            clear_gpu_cache(local_rank)
    arguments = {
        "rank": rank if conf.enable_fsdp else None,
        "device": device if not conf.enable_fsdp else None
        }
    tokenizer, model, optmizer = load_model(conf, **arguments)
    pdb.set_trace()
    save_model(model, optmizer, conf, 0, rank)