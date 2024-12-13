from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import(
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing
)

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

# Check the layer exists
check_fn = lambda submodule: isinstance(submodule, [LlamaDecoderLayer,
                                                    MixtralAttention,
                                                    Qwen2DecoderLayer,
                                                    MistralDecoderLayer])


# Usage of non-reentrant style for the best performance
non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    offload_to_cpu=False,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT
)


# Send a model to this func. after wrapping to FSDP.
# pros: Free up 35-40% of GPU, cons: Slow down training 20-25%
def apply_fsdp_checkpointing(model):

    print(f"--> applying fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model=model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=check_fn
    )