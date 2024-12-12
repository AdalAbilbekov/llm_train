from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import pdb
from omegaconf import OmegaConf

import torch
import torch.optim as optim
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from policies import get_policies

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp import ShardingStrategy
import torch.distributed as dist
import os

def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")

# TODO: Add wraping policy for sharding model via auto_wrap_policy and FSDP
def set_model(model, model_config, rank):
    if model_config.enable_fsdp:
        mixed_precision_policy, wrapping_policy = get_policies(model_config, rank)
        model = FSDP(
            model,
            auto_wrap_policy= wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if model_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=False,
            param_init_fn=None,
            )
        pdb.set_trace()
        pass
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

def load_optimizer(model, model_config):
    return optim.AdamW(
        model.parameters(),
        lr=model_config.lr,
        # weight_decay=model_config.weight_decay
    )

# TODO: add model parallel sharding.
def load_model(model_config):
    model = load(model_config)
    setup()
    rank = int(os.environ["RANK"])
    model = set_model(model, model_config, rank)

    tokenizer = AutoTokenizer.from_pretrained(model_config.model.path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    optimizer = load_optimizer(model, model_config)

    return tokenizer, model, optimizer

    
if __name__=="__main__":
    conf = OmegaConf.load("config.yaml")
    tokenizer, model, optmizer = load_model(conf)