import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
import pdb
import functools

from pkg_resources import packaging
from policies import fpSixteen
from  transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from  transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)

import functools
from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

def get_wrapper():
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer, # <- Deocder layer to apply FSDP sharding policy.
            GPTNeoXLayer,
            MistralDecoderLayer,
            MixtralDecoderLayer
        },
    )

    return auto_wrap_policy