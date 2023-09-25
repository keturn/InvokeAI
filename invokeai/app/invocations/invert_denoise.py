from typing import List, Union

import torch
from diffusers import AutoencoderKL, DDIMInverseScheduler, DDIMScheduler, DDPMScheduler, \
    DPMSolverMultistepInverseScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, \
    StableDiffusionPix2PixZeroPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import BlipForConditionalGeneration, BlipProcessor, CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    invocation,
)
from invokeai.app.invocations.model import UNetField
from invokeai.app.invocations.primitives import ConditioningField, LatentsField, LatentsOutput, build_latents_output
from invokeai.app.util.misc import SEED_MAX, get_random_seed


@invocation(
    "invert_denoise_latents",
    title="Invert Denoise Latents",
    tags=["latents", "denoise"],
    category="latents",
    version="0.1.0",
)
class InvertDenoiseLatentsInvocation(BaseInvocation):
    positive_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection, ui_order=0
    )
    negative_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.negative_cond, input=Input.Connection, ui_order=1
    )
    steps: int = InputField(default=10, gt=0, description=FieldDescriptions.steps)
    cfg_scale: Union[float, List[float]] = InputField(
        default=7.5, ge=1, description=FieldDescriptions.cfg_scale, title="CFG Scale"
    )
    # denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    # denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    unet: UNetField = InputField(description=FieldDescriptions.unet, input=Input.Connection, title="UNet", ui_order=2)
    latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    seed: int = InputField(
        ge=0,
        le=SEED_MAX,
        description=FieldDescriptions.seed,
        default_factory=get_random_seed,
    )

    def _create_pipeline(self, unet, scheduler):
        return LesserInvertPipeline(
            vae=None,
            unet=unet,
            text_encoder=None,
            tokenizer=None,
            inverse_scheduler=scheduler,
            requires_safety_checker=False,
        )

    def _get_scheduler(self, context):
        # derived from invocations.latent.get_scheduler
        scheduler_info = self.unet.scheduler
        orig_scheduler_info = context.services.model_manager.get_model(
            **scheduler_info.dict(),
            context=context,
        )
        with orig_scheduler_info as orig_scheduler:
            scheduler_config = orig_scheduler.config

        if "_backup" in scheduler_config:
            scheduler_config = scheduler_config["_backup"]
        scheduler_config = {
            **scheduler_config,
            "_backup": scheduler_config,
        }

        return DPMSolverMultistepInverseScheduler.from_config(scheduler_config)

    def _get_prompt_embeds(self, context: InvocationContext, device: torch.device) -> (torch.FloatTensor, torch.FloatTensor):
        positive_cond_data = context.services.latents.get(self.positive_conditioning.conditioning_name)
        c = positive_cond_data.conditionings[0].embeds.to(device=device)

        negative_cond_data = context.services.latents.get(self.negative_conditioning.conditioning_name)
        uc = negative_cond_data.conditionings[0].embeds.to(device=device)

        return c, uc

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        unet_info = context.services.model_manager.get_model(
            **self.unet.unet.dict(),
            context=context,
        )
        latents = context.services.latents.get(self.latents.latents_name)
        with unet_info as unet:
            device = unet.device
            latents = latents.to(device)
            prompt_embeds, negative_prompt_embeds = self._get_prompt_embeds(context, device)
            scheduler = self._get_scheduler(context)
            pipeline = self._create_pipeline(unet, scheduler)
            # Why is there RNG in here? It's used in the `auto_corr_loss` method.
            generator = torch.Generator(device='cpu').manual_seed(self.seed)
            result_latents = pipeline.lesser_invert(
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=self.steps,
                guidance_scale=self.cfg_scale,
                generator=generator,
            )

        name = f"{context.graph_execution_state_id}__{self.id}"
        context.services.latents.save(name, result_latents)
        return build_latents_output(latents_name=name, latents=result_latents)



class FakeVae:
    class FakeVaeConfig:
        def __init__(self):
            self.block_out_channels = [0]

    def __init__(self):
        self.config = FakeVae.FakeVaeConfig()

class LesserInvertPipeline(StableDiffusionPix2PixZeroPipeline):
    """I just wanted to use the invert method."""

    def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: Union[DDPMScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler] = None,
                 feature_extractor: CLIPImageProcessor = None, safety_checker: StableDiffusionSafetyChecker = None,
                 inverse_scheduler: DDIMInverseScheduler = None, caption_generator: BlipForConditionalGeneration = None,
                 caption_processor: BlipProcessor = None, requires_safety_checker: bool = False):
        if vae is None:
            vae = FakeVae()
            vae.dtype = unet.dtype

        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, feature_extractor, safety_checker, inverse_scheduler, caption_generator, caption_processor, requires_safety_checker)

    @torch.inference_mode()
    def lesser_invert(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        lambda_auto_corr: float = 20.0,
        lambda_kl: float = 20.0,
        num_reg_steps: int = 5,
        num_auto_corr_rolls: int = 5,
    ) -> torch.Tensor:
        # I tried to use upstream code, but it wasn't granular enough. Much of this is copypasta
        # from the upstream invert method, which is itself mostly identical to other pipeline methods.

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps

        # 7. Denoising loop where we obtain the cross-attention maps.
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.inverse_scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # regularization of the noise prediction
            with torch.inference_mode(False), torch.enable_grad():
                for _ in range(num_reg_steps):
                    if lambda_auto_corr > 0:
                        for _ in range(num_auto_corr_rolls):
                            var = torch.autograd.Variable(noise_pred.detach().clone(), requires_grad=True)

                            # Derive epsilon from model output before regularizing to IID standard normal
                            var_epsilon = self.get_epsilon(var, latent_model_input.detach(), t)

                            l_ac = self.auto_corr_loss(var_epsilon, generator=generator)
                            l_ac.backward()

                            grad = var.grad.detach() / num_auto_corr_rolls
                            noise_pred = noise_pred - lambda_auto_corr * grad

                    if lambda_kl > 0:
                        var = torch.autograd.Variable(noise_pred.detach().clone(), requires_grad=True)

                        # Derive epsilon from model output before regularizing to IID standard normal
                        var_epsilon = self.get_epsilon(var, latent_model_input.detach(), t)

                        l_kld = self.kl_divergence(var_epsilon)
                        l_kld.backward()

                        grad = var.grad.detach()
                        noise_pred = noise_pred - lambda_kl * grad

                    noise_pred = noise_pred.detach()

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample

        inverted_latents = latents.detach().clone()

        return inverted_latents
