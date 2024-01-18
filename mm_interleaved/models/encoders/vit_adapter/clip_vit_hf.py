import collections.abc
from itertools import repeat
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPEncoder, CLIPPreTrainedModel, CLIP_VISION_INPUTS_DOCSTRING, \
    CLIP_START_DOCSTRING
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

from .xattn import convert_clip_visual_attn


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    
    return parse


to_2tuple = _ntuple(2)


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))
    
    def interpolate_pos_embed(self, image_size):
        if image_size == self.image_size:
            return
        
        num_patches = (image_size // self.patch_size) ** 2
        print(f"interpolate CLIP image pos embed from {self.num_patches} to {num_patches}")
        
        old_pos_embed = self.position_embedding.weight
        extra_tokens = 1
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
        
        grid_size = to_2tuple(int(num_patches ** 0.5))
        old_grid_size = int(self.num_patches ** 0.5)
        pos_emb_img = pos_emb_img.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
        pos_emb_img = F.interpolate(
            pos_emb_img,
            size=grid_size,
            mode='bicubic',
            align_corners=True,
        )
        pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, num_patches, -1)[0]
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
        
        self.image_size = image_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim, _weight=new_pos_embed)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        Hp, Wp = patch_embeds.shape[2], patch_embeds.shape[3]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings, Hp, Wp


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        # self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.post_layernorm = nn.Identity()
    
    def interpolate_pos_embed(self, image_size):
        self.embeddings.interpolate_pos_embed(image_size)
    
    def set_vis_embed_requires_grad(self, requires_grad):
        self.vis_embed_requires_grad = requires_grad
    
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        
        hidden_states, _, _ = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        
        # hidden_states.requires_grad_(self.vis_embed_requires_grad)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)
        
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """The vision model from CLIP without any head or projection on top.""",
    CLIP_START_DOCSTRING,
)
class CLIPVisionModel(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"
    
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()
    
    def interpolate_pos_embed(self, image_size):
        self.vision_model.interpolate_pos_embed(image_size)
    
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding
    
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPVisionModel

        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


def clip_vit_hf(**kwargs):
    freeze = kwargs.pop('freeze', False)
    image_size = kwargs.pop('image_size', 224)
    model_path = kwargs.pop('model_path', "openai/clip-vit-large-patch14")
    model = CLIPVisionModel.from_pretrained(model_path)
    gradient_checkpointing = kwargs.pop('gradient_checkpointing', False)
    model.vision_model.encoder.gradient_checkpointing = gradient_checkpointing
    model.interpolate_pos_embed(image_size)
    # NOTE we do not use pooler output
    model.vision_model.post_layernorm.requires_grad_(False)
    convert_clip_visual_attn(model)
    print(f"Freeze clip_vit_hf is {freeze}")
    model.requires_grad_((not freeze))
    freeze_stem = kwargs.pop('freeze_stem', freeze)
    print(f"Freeze clip_vit_hf stem is {freeze_stem}")
    if freeze_stem:
        model.vision_model.embeddings.requires_grad_(False)
        model.vision_model.pre_layrnorm.requires_grad_(False)
    model.vision_model.set_vis_embed_requires_grad((not freeze))
    
    return model
