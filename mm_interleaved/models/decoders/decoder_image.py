import torch
import torch.nn as nn
from einops import rearrange

from .perceiver import PerceiverResampler
from .sd import StableDiffusion


class ImageDecoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path="",
        uncond_prob=0.1,
        seq_len=77,
        embed_dim=1024,
        image_size=512,
        mmfs_input_channel=1024,
        mmfs_feat_levels=4,
        vae_encode_mini_bs=32,
        sd_base_seed=0,
        sd_use_random_seed=False,
        sd_use_vae_gradient_checkpointing=True,
        sd_use_unet_gradient_checkpointing=True,
        perceiver_config=None,
    ):
        super().__init__()
        self.uncond_prob = uncond_prob
        
        self.perceiver_resampler = PerceiverResampler(**perceiver_config)
        self.decoder = StableDiffusion(
            pretrained_model_name_or_path,
            image_size=image_size,
            use_vae_gradient_checkpointing=sd_use_vae_gradient_checkpointing,
            use_unet_gradient_checkpointing=sd_use_unet_gradient_checkpointing,
            vae_encode_mini_bs=vae_encode_mini_bs,
            base_seed=sd_base_seed,
            use_random_seed=sd_use_random_seed,
            mmfs_input_channel=mmfs_input_channel,
            mmfs_feat_levels=mmfs_feat_levels,
        )

        if self.uncond_prob > 0:
            self.neg_prompt_embeds = nn.Parameter(
                torch.zeros(1, seq_len, embed_dim)
            )
            nn.init.normal_(self.neg_prompt_embeds, std=0.02)
            assert self.neg_prompt_embeds.shape[1] == seq_len
            neg_prompt_embeds = self.decoder.get_negative_prompt_embeds(
                uncond_tokens=[""],
                device="cuda",
                dtype=self.neg_prompt_embeds.dtype,
            )
            neg_prompt_embeds = neg_prompt_embeds.to(
                device=self.neg_prompt_embeds.device
            )
            self.neg_prompt_embeds.data.copy_(neg_prompt_embeds)

    def print_parameters_stats(self, prefix=""):
        for name, module in self.named_children():
            print(
                f"# {prefix}{name} Total parameters: {sum(p.numel() for p in module.parameters()) / 1e6:.2f}M"
            )
            print(
                f"# {prefix}{name} Trainable parameters: {sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6:.2f}M"
            )
            if hasattr(module, "print_parameters_stats"):
                module.print_parameters_stats(prefix=f"{prefix}{name}.")

    def forward(
        self,
        image_tensors,
        context_features,
        context_attention_mask=None,
        image_loss_mask=None,
        mmfs_features=None,
        mmfs_mask=None,
        **kwargs,
    ):
        """
        image_tensors: [B_I, 3, H, W]
        context_features: [B_I, L, D]
        """
        assert image_tensors.shape[0] == context_features.shape[0]
        if context_attention_mask is not None:
            assert torch.all(
                context_attention_mask.sum(dim=1) > 0
            ), f"{context_attention_mask.sum(dim=1)=}"
        
        context_features = self.perceiver_resampler(
            encoder_hidden_states=context_features,
            encoder_attention_mask=context_attention_mask,
            return_dict=False,
        )[0]
        
        if self.uncond_prob > 0.0:
            uncond_mask = (
                torch.rand_like(context_features[:, :1, :1]) < self.uncond_prob
            )
            neg_prompt_embeds = self.neg_prompt_embeds
            context_features = torch.where(
                uncond_mask, neg_prompt_embeds, context_features
            )
        
        sd_loss = self.decoder(
            image_tensors,
            context_features,
            mmfs_features=mmfs_features,
            mmfs_mask=mmfs_mask,
            **kwargs,
        )
        assert context_attention_mask is not None
        is_cond_image = context_attention_mask.sum(dim=1) > 2  # [<bos>, <soi>]
        is_cond_image = rearrange(is_cond_image, "b -> b 1 1 1")
        sd_loss = sd_loss * is_cond_image
        if image_loss_mask is not None:
            image_loss_mask = rearrange(image_loss_mask, "b -> b 1 1 1")
            sd_loss = sd_loss * image_loss_mask
        sd_loss = sd_loss.mean()

        return sd_loss

    @torch.no_grad()
    def generate_images(
        self,
        context_features,
        context_attention_mask=None,
        mmfs_features=None,
        mmfs_mask=None,
        **kwargs,
    ):
        output = {}

        context_features = self.perceiver_resampler(
            encoder_hidden_states=context_features,
            encoder_attention_mask=context_attention_mask,
            return_dict=False,
        )[0]
        num_inference_steps = kwargs.pop("num_inference_steps", 30)
        guidance_scale = kwargs.pop("guidance_scale", 7.5)
        num_validation_images = kwargs.pop("num_validation_images", 1)

        negative_prompt_embeds = self.neg_prompt_embeds.expand_as(
            context_features
        )
        images = self.decoder.generate_images(
            text_embeds=context_features,
            negative_prompt_embeds=negative_prompt_embeds,
            num_validation_images=num_validation_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            mmfs_features=mmfs_features,
            mmfs_mask=mmfs_mask,
        )

        output["image"] = images
        return output
