from dataclasses import dataclass
from pathlib import Path

import PIL.Image
import pytest
import torch
from diffusers.utils import BaseOutput

from invokeai.backend.stable_diffusion.diffusers_pipeline import ColorizingGuidance, PredictsOriginalOutput


@pytest.fixture
def bw_photo(datadir: Path) -> PIL.Image.Image:
    # cropped and resized from https://lccn.loc.gov/2020635851
    return PIL.Image.open(datadir / "restaurant-counter-squarecrop.png")


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:0")


@pytest.fixture
def rng(device: torch.device) -> torch.Generator:
    generator = torch.Generator(device)
    generator.seed()
    return generator


def test_creation_from_image(bw_photo: PIL.Image.Image, device):
    colorizer = ColorizingGuidance.from_image(bw_photo, device=device, dtype=torch.float16)
    original_image_values = colorizer.original_value_channel
    assert original_image_values.size() == (1, 1, 512, 512)
    assert original_image_values.device == device
    assert 0 <= original_image_values.min() < 1
    assert 0 < original_image_values.max() <= 1


def test_latents_to_value_channel(device, rng):
    colorizer = ColorizingGuidance(torch.rand((1, 1, 512, 512), device=device, generator=rng))
    latents = torch.rand((1, 4, 64, 64), device=device, generator=rng, dtype=torch.float16)
    value_channel = colorizer.latents_to_luv(latents)
    assert 0 <= value_channel.min() < 1
    assert 0 < value_channel.max() <= 1


def test_value_loss(bw_photo: PIL.Image.Image, rng: torch.Generator, device: torch.device):
    colorizer = ColorizingGuidance.from_image(bw_photo, device=device, dtype=torch.float16)
    latents = torch.rand((1, 4, 64, 64), device=device, generator=rng, dtype=torch.float16)
    loss = colorizer.chroma_loss(latents)
    assert loss > 0


@dataclass
class DummyStepOutput(BaseOutput):
    prev_sample: torch.FloatTensor
    pred_original_sample: torch.FloatTensor


def test_grad(bw_photo: PIL.Image.Image, rng: torch.Generator, device: torch.device):
    with torch.inference_mode():
        colorizer = ColorizingGuidance.from_image(bw_photo, device=device, dtype=torch.float16)
        prev_sample = torch.rand((1, 4, 64, 64), device=device, generator=rng, dtype=torch.float16)
        pred_original_sample = torch.rand((1, 4, 64, 64), device=device, generator=rng, dtype=torch.float16)
        step_output: PredictsOriginalOutput = DummyStepOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )
        grad = colorizer.find_conditioning_gradient(step_output)
        assert grad.size() == (1, 4, 64, 64)
        assert not grad.isnan().any()


def test_step(bw_photo: PIL.Image.Image, rng: torch.Generator, device: torch.device):
    with torch.inference_mode():
        colorizer = ColorizingGuidance.from_image(bw_photo, device=device, dtype=torch.float16)
        prev_sample = torch.rand((1, 4, 64, 64), device=device, generator=rng, dtype=torch.float16)
        pred_original_sample = torch.rand((1, 4, 64, 64), device=device, generator=rng, dtype=torch.float16)
        step_output = DummyStepOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
        colorizer(step_output, None, None)
