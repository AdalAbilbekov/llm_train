import torch.distributed as dist
import torch

def setup():
    dist.init_process_group("nccl")

def clear_gpu_cache(rank=None):
    if rank==0:
        print("Clearing the GPU cache on all GPUs")
    torch.cuda.empty_cache()
