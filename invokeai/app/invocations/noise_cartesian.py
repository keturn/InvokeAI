# Copyright 2023 Kevin Turner
# SPDX-License-Identifier: Apache-2.0
"""Cartesian Noise Generator.

required dependencies: cityhash~=0.4 pymorton~=1.0
"""

import secrets
import struct
from typing import Union

import farmhash
import numpy as np
import pymorton
import scipy
import torch

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    FieldDescriptions,
    InputField,
    InvocationContext,
    invocation,
)
from invokeai.app.invocations.noise import NoiseOutput, build_noise_output
from invokeai.app.util.misc import get_random_seed
from invokeai.backend.util import choose_torch_device, torch_dtype

SEED_MAX = 0xFFFFFFFF  # FarmHash32 seed is uint32

int_packing = struct.Struct("<i")


def morton_fill(position: np.ndarray, shape: Union[tuple, np.ndarray], seed: int = None) -> np.ndarray:
    if seed is None:
        seed = secrets.randbits(32)
    a = np.empty(shape, np.uint32)

    # we've imported these optimized functions in pymorton and farmhash
    # and then make them slow because pymorton doesn't work on numpy data types
    # and farmhash doesn't work on ints

    buf = bytearray(4)
    with np.nditer(a, flags=["multi_index"], op_flags=["writeonly"]) as it:
        for x in it:
            # Use morton encoding to convert our multidimensional coordinate to a single value to hash.
            # Is morton encoding better than just packing the inputs to sequential bytes? It shouldn't if
            # the following hash is good, but it seems like it would help resist streaks in rows or columns.
            point = position + it.multi_index
            z_order = pymorton.interleave3(*point.tolist())
            # convert the integer to bytes we can feed to the hash algorithm
            int_packing.pack_into(buf, 0, z_order)
            # I remember putting doing a fair bit of looking over various options before picking
            # farmhash, but do I remember why now? https://github.com/google/farmhash has since been
            # put in to read-only archive mode which makes this even more questionable.
            # The hash algorithm needn't be cryptographically secure, but we want something that we
            # can use on these inputs that are relatively small and close together, and we don't want
            # to see streaks or other patterns in the results even when we're only looking at 16 bits
            # of the result.
            x[...] = farmhash.FarmHash32WithSeed(buf, seed)

    # Convert unsigned int32 uniform distribution to normal distribution
    return scipy.stats.norm.ppf(a / (1 << 32), 0).astype(np.float16)


@invocation("noise_cartesian", title="Cartesian Noise", tags=["latents", "noise"], category="latents", version="1.0.0")
class CartesianNoiseInvocation(BaseInvocation):
    """Generates latent noise."""

    _downsampling_factor = 8
    _latent_channels = 4

    seed: int = InputField(
        ge=0,
        le=SEED_MAX,
        description=FieldDescriptions.seed,
        default_factory=get_random_seed,
    )
    width: int = InputField(
        default=512,
        multiple_of=_downsampling_factor,
        gt=0,
        description=FieldDescriptions.width,
    )
    height: int = InputField(
        default=512,
        multiple_of=_downsampling_factor,
        gt=0,
        description=FieldDescriptions.height,
    )
    x0: int = InputField(default=0, description="x-coordinate of the lower edge")
    y0: int = InputField(default=0, description="y-coordinate of the lower edge")
    z0: int = InputField(default=0, description="coordinate of the first channel")

    def invoke(self, context: InvocationContext) -> NoiseOutput:
        origin = np.array([self.x0, self.y0, self.z0])
        origin //= self._downsampling_factor
        shape = np.array(
            [self._latent_channels, self.width // self._downsampling_factor, self.height // self._downsampling_factor]
        )
        np_noise = morton_fill(origin, shape, self.seed)

        noise_tensor = torch.tensor(np_noise, dtype=torch_dtype(choose_torch_device()))
        noise_tensor.unsqueeze_(0)  # batch size of 1

        name = f"{context.graph_execution_state_id}__{self.id}"
        context.services.latents.save(name, noise_tensor)
        return build_noise_output(latents_name=name, latents=noise_tensor, seed=self.seed)
