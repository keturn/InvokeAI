# from https://arxiv.org/pdf/2506.19713

import kornia.geometry
import torch
from kornia.geometry.transform import build_laplacian_pyramid


def project(
    v0: torch.Tensor, # [B, C, H, W]
    v1: torch.Tensor, # [B, C, H, W]
    ):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

def build_image_from_pyramid(pyramid, shape=None):
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        img = kornia.geometry.pyrup(img) + pyramid[i]
    if shape is not None:
        # remove padding added by build_laplacian_pyramid
        img = img[:, :, :shape[-2], :shape[-1]]
    return img

# We assume all model predictions are converted to "x_0" prediction.
def laplacian_guidance(
    pred_cond: torch.Tensor, # [B, C, H, W]
    pred_uncond: torch.Tensor, # [B, C, H, W]
    guidance_scale=(1.0, 1.0), # Guidance scales from high- to low-frequency
    parallel_weights=None, # Optional weights for projection
    ):
    levels = len(guidance_scale)
    if parallel_weights is None:
        parallel_weights = [1.0] * levels
    pred_cond_pyramid = build_laplacian_pyramid(pred_cond, levels)
    pred_uncond_pyramid = build_laplacian_pyramid(pred_uncond, levels)
    pred_guided_pyramid = []
    parameters = zip(pred_cond_pyramid, pred_uncond_pyramid, guidance_scale, parallel_weights, strict=False)
    for p_cond, p_uncond, scale, par_weight in parameters:
        diff = p_cond - p_uncond
        diff_parallel, diff_orthogonal = project(diff, p_cond)
        diff = par_weight * diff_parallel + diff_orthogonal
        p_guided = p_cond + (scale - 1) * diff
        pred_guided_pyramid.append(p_guided)
    pred_guided = build_image_from_pyramid(pred_guided_pyramid, pred_cond.shape)
    return pred_guided.to(pred_cond.dtype)
