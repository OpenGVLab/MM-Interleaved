import os
import numpy as np
import math
import warnings

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F


from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from diffusers.utils.import_utils import is_xformers_available

from .sd_mmfs import MMFSNet


class StableDiffusion(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path="",
        revision=None,
        noise_offset=0.0,
        image_size=512,
        use_vae_gradient_checkpointing=True,
        use_unet_gradient_checkpointing=True,
        vae_encode_mini_bs=32,
        base_seed=0,
        use_random_seed=False,
        mmfs_input_channel=1024,
        mmfs_feat_levels=4,
    ) -> None:
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.revision = revision
        self.noise_offset = noise_offset
        self.image_size = image_size
        self.vae_encode_mini_bs = vae_encode_mini_bs
        self.base_seed = base_seed
        self.use_random_seed = use_random_seed

        self.noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )

        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", revision=revision
        )

        self.vae.requires_grad_(False)
        for module in self.vae.modules():
            self.vae._set_gradient_checkpointing(module, use_vae_gradient_checkpointing)

        unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", revision=revision
        )

        assert is_xformers_available()
        unet.enable_xformers_memory_efficient_attention()
        
        self.unet = unet
        for module in self.unet.modules():
            self.unet._set_gradient_checkpointing(
                module, use_unet_gradient_checkpointing
            )

        config = self.unet.config
        self.mmfs_module = MMFSNet(
            input_channel=mmfs_input_channel,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            downsample_factor=512 // self.image_size,
            n_levels=mmfs_feat_levels,
            gradient_checkpointing=use_unet_gradient_checkpointing,
        )

        self.print_trainable_parameters()

    def print_parameters_stats(self, prefix=""):
        for name, module in self.named_children():
            print(
                f"# {prefix}{name} Total parameters: {sum(p.numel() for p in module.parameters()) / 1e6:.2f}M"
            )
            print(
                f"# {prefix}{name} Trainable parameters: {sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6:.2f}M"
            )

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    @torch.no_grad()
    def get_negative_prompt_embeds(
        self, uncond_tokens=[""], device="cuda", dtype=torch.float16
    ):
        pipeline = DiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            unet=self.unet,
            revision=self.revision,
            scheduler=self.noise_scheduler,
        )
        pipeline = pipeline.to(device, dtype)

        uncond_input = pipeline.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        negative_prompt_embeds = pipeline.text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=None,
        )["last_hidden_state"]
        negative_prompt_embeds = negative_prompt_embeds.detach()
        print(f"{negative_prompt_embeds.shape=} {negative_prompt_embeds.device=}")

        return negative_prompt_embeds

    @torch.no_grad()
    def generate_images(
        self,
        text_embeds,
        negative_prompt_embeds=None,
        num_validation_images=1,
        num_inference_steps=30,
        mini_bs=8,
        guidance_scale=7.5,
        mmfs_features=None,
        mmfs_mask=None,
    ):
        pipeline = DiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            unet=self.unet,
            revision=self.revision,
            scheduler=self.noise_scheduler,
        )
        pipeline.set_progress_bar_config(disable=True)
        pipeline = pipeline.to(text_embeds.device, text_embeds.dtype)
        pipeline.vae = pipeline.vae.float()
        images = []
        for num in range(num_validation_images):
            if self.use_random_seed:
                seed = num + np.random.randint(self.base_seed)
            else:
                seed = num + self.base_seed
            generator = torch.Generator(device=text_embeds.device).manual_seed(seed)
            for mini_iter in range(math.ceil(text_embeds.shape[0] / mini_bs)):
                txt_emb = text_embeds[
                    mini_iter * mini_bs : mini_iter * mini_bs + mini_bs
                ]
                neg_emb = (
                    negative_prompt_embeds[
                        mini_iter * mini_bs : mini_iter * mini_bs + mini_bs
                    ]
                    if negative_prompt_embeds is not None
                    else None
                )

                ms_ctrl_mask = (
                    mmfs_mask[
                        mini_iter * mini_bs : mini_iter * mini_bs + mini_bs
                    ]
                    if mmfs_mask is not None
                    else None
                )
                ms_ctrl_feats = (
                    [
                        ms_feat[mini_iter * mini_bs : mini_iter * mini_bs + mini_bs]
                        for ms_feat in mmfs_features
                    ]
                    if mmfs_features is not None
                    else None
                )

                image = pipeline(
                    prompt_embeds=txt_emb,
                    negative_prompt_embeds=neg_emb,
                    width=self.image_size,
                    height=self.image_size,
                    output_type="latent",
                    cross_attention_kwargs=None,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    mmfs_features=ms_ctrl_feats,
                    mmfs_mask=ms_ctrl_mask,
                    mmfs_module=self.mmfs_module,
                ).images
                latents = 1 / pipeline.vae.config.scaling_factor * image
                latents = latents.float()
                image = pipeline.vae.decode(latents, return_dict=False)[0]
                image = (image / 2 + 0.5).clamp(0, 1).float().detach()
                images.append(image)
        images = torch.cat(images, dim=0)
        return images

    @torch.no_grad()
    def _encode_latents(self, image):
        dtype = image.dtype
        self.vae = self.vae.float()

        mini_bs = self.vae_encode_mini_bs
        if mini_bs > 0:
            latents = []
            for mini_iter in range(math.ceil(image.shape[0] / mini_bs)):
                _image = image[mini_iter * mini_bs : mini_iter * mini_bs + mini_bs]
                _image = _image.float()
                _latents = self.vae.encode(_image).latent_dist.sample()
                latents.append(_latents.to(dtype))
            latents = torch.cat(latents, dim=0)
        else:
            latents = self.vae.encode(image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        return latents

    def forward(
        self,
        image,
        text_embeds,
        return_outputs=False,
        mmfs_features=None,
        mmfs_mask=None,
    ):
        h, w = image.shape[-2:]
        assert (
            h == self.image_size and w == self.image_size
        ), f"{image.shape=} {self.image_size=}"
        if h != self.vae.config.sample_size or w != self.vae.config.sample_size:
            warnings.warn(
                f"The input image size {h} * {w} is not equal to the sample size {self.vae.config.sample_size} of vae model"
            )

        # normalize image manually
        image = image.sub_(0.5).div_(0.5)

        latents = self._encode_latents(image)

        with torch.no_grad():
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            if self.noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += self.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                )

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        # encoder_hidden_states = text_encoder(batch["input_ids"])[0]
        encoder_hidden_states = text_embeds

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        # Predict the noise residual and compute loss
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            cross_attention_kwargs=None,
            mmfs_features=mmfs_features,
            mmfs_mask=mmfs_mask,
            mmfs_module=self.mmfs_module,
        ).sample
        
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")

        return (
            loss
            if not return_outputs
            else dict(loss=loss, pred=model_pred, target=target)
        )
