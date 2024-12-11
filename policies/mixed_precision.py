import torch

from torch.distributed.fsdp import (
    MixedPrecision,
)

# TODO: Add brain float 16 precision.
# requires grad scaler in main loop
fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)