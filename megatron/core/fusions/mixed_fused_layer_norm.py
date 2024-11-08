# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""This code is copied from NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

import numbers
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import importlib
from torch.nn import functional as F
import inspect

from megatron.core.utils import make_viewless_tensor

from megatron.core.transformer import TransformerConfig

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN

    HAVE_PERSIST_LAYER_NORM = True
except:
    HAVE_PERSIST_LAYER_NORM = False

from apex.normalization.fused_layer_norm import (
    FusedLayerNormAffineFunction,
    FusedRMSNormAffineFunction,
)


global fused_layer_norm_cuda
fused_layer_norm_cuda = None

class MixedFusedRMSNorm(torch.nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
    ):
        super(MixedFusedRMSNorm, self).__init__()

        self.layernorm_zero_centered_gamma = self.config.layernorm_zero_centered_gamma
        self.memory_efficient_layer_norm = self.config.memory_efficient_layer_norm
        self.norm_fn = FusedRMSNormAffineFunction

        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        no_persist_layer_norm = not self.config.persist_layer_norm
        persist_ln_hidden_sizes = [
            1024,
            1536,
            2048,
            2304,
            3072,
            3840,
            4096,
            5120,
            6144,
            8192,
            10240,
            12288,
            12800,
            15360,
            16384,
            18432,
            20480,
            24576,
            25600,
            30720,
            32768,
            40960,
            49152,
            65536,
        ]
        if (
            hidden_size not in persist_ln_hidden_sizes
            or not HAVE_PERSIST_LAYER_NORM
        ):
            no_persist_layer_norm = True

        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        self.normalized_shape = torch.Size(hidden_size)
        self.eps = eps
        self.scale = Parameter(torch.Tensor(*hidden_size))
        self.reset_parameters()
        self.no_persist_layer_norm = no_persist_layer_norm
        self.sequence_parallel = self.config.sequence_parallel

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.scale, "sequence_parallel", self.sequence_parallel)

    def reset_parameters(self):

        if self.layernorm_zero_centered_gamma:
            init.zeros_(self.scale)
        else:
            init.ones_(self.scale)

    def forward(self, input):

        weight = self.scale + 1 if self.layernorm_zero_centered_gamma else self.scale
        # CPU path is here for unittest sake.
        if not input.is_cuda:
            print(
                "WARNING! The input of FusedLayerNorm should be on the GPU."
                "This warning should only be triggered in the FusedRMSNorm unit tests."
            )
            # Latest pytorch actually supports F.rms_norm but I don't want to break builds so...
            return F.layer_norm(input, self.normalized_shape, weight, None, self.eps)

        # Apex does not have versions yet (https://github.com/NVIDIA/apex/pull/1648), so we need to inspect
        # the function manually on whether the extra arg introduced in https://github.com/NVIDIA/apex/pull/1715 exists yet
        if "memory_efficient" in inspect.getfullargspec(self.norm_fn.forward).args:
            return self.norm_fn.apply(
                input,
                weight,
                self.normalized_shape,
                self.eps,
                self.memory_efficient_layer_norm,
            )
        else:
            return self.norm_fn.apply(input, weight, self.normalized_shape, self.eps)

            # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
            # a populated '_base' field). This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            output = make_viewless_tensor(
                inp=output, requires_grad=input.requires_grad, keep_graph=True
            )

            return output
