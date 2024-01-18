import collections.abc
from itertools import repeat
from typing import Optional, Tuple, Union

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
from .adapter_modules import SpatialPriorModule, InteractionBlockWithCls
from .adapter_modules import deform_inputs
from .clip_vit_hf import CLIPVisionEmbeddings
from .ops.modules import MSDeformAttn


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    
    return parse


to_2tuple = _ntuple(2)


class CLIPVisionTransformerAdapter(nn.Module):
    def __init__(self, config: CLIPVisionConfig, conv_inplane=64, n_points=4):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        image_size = config.image_size
        num_attention_heads = config.num_attention_heads
        num_hidden_layers = config.num_hidden_layers
        
        if num_hidden_layers == 24:  # for ViT-Large
            self.interaction_indexes = [[0, 5], [6, 11], [12, 17], [18, 23]]
        else:
            raise NotImplementedError
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        
        # adapter modules
        self.adapter_level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.adapter_spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.adapter_interactions = nn.Sequential(*[
            InteractionBlockWithCls(
                dim=embed_dim, num_heads=num_attention_heads, n_points=n_points,
                init_values=0., drop_path=0., norm_layer=nn.LayerNorm, with_cffn=True,
                cffn_ratio=0.25, deform_ratio=0.5, with_cp=True,
                extra_extractor=True if i == len(self.interaction_indexes) - 1 else False)
            for i in range(len(self.interaction_indexes))
        ])
        self.adapter_up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        
        # self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.post_layernorm = nn.Identity()
    
    def interpolate_pos_embed(self, image_size):
        self.embeddings.interpolate_pos_embed(image_size)
    
    def set_vis_embed_requires_grad(self, requires_grad):
        self.vis_embed_requires_grad = requires_grad
    
    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()
    
    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.adapter_level_embed[0]
        c3 = c3 + self.adapter_level_embed[1]
        c4 = c4 + self.adapter_level_embed[2]
        return c2, c3, c4
    
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
        
        hidden_states, H, W = self.embeddings(pixel_values)
        bs, n, dim = hidden_states.shape
        hidden_states = self.pre_layrnorm(hidden_states)
        
        new_size = self.config.image_size // self.config.patch_size * 16
        pixel_values_resized = F.interpolate(pixel_values, size=(new_size, new_size), mode="bilinear", align_corners=False)
        deform_inputs1, deform_inputs2 = deform_inputs(pixel_values_resized)
        
        # SPM forward
        c1, c2, c3, c4 = self.adapter_spm(pixel_values_resized)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)
        
        hidden_states.requires_grad_(self.vis_embed_requires_grad)
        # Interaction
        x = hidden_states[:, 1:, :]
        cls = hidden_states[:, 0:1, :]
        encoder_layers = self.encoder.layers
        
        outs = list()
        for i, layer in enumerate(self.adapter_interactions):
            indexes = self.interaction_indexes[i]
            self.encoder.layers = encoder_layers[indexes[0]:indexes[-1] + 1]
            x, c, cls = layer(x, c, cls, self.encoder,
                              deform_inputs1, deform_inputs2, H, W)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())
        self.encoder.layers = encoder_layers
        
        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]
        
        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.adapter_up(c2) + c1
        
        x1, x2, x3, x4 = outs
        last_hidden_state = x4.flatten(2).transpose(1, 2)
        x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
        c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4
        multiscale_features = [c1, c2, c3, c4]
        
        last_hidden_state = torch.cat([cls, last_hidden_state], dim=1)
        pooled_output = cls
        pooled_output = self.post_layernorm(pooled_output)
        
        if not return_dict:
            return (last_hidden_state, pooled_output) + multiscale_features
        
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=multiscale_features,
            attentions=None
        )


@add_start_docstrings(
    """The vision model from CLIP without any head or projection on top.""",
    CLIP_START_DOCSTRING,
)
class CLIPVisionAdapterModel(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"
    
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionTransformerAdapter(config)
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


def clip_vit_adapter_hf(
    model_path="openai/clip-vit-large-patch14",
    image_size=224,
    freeze=False,
    freeze_vit=True,
    gradient_checkpointing=True,
):
    model = CLIPVisionAdapterModel.from_pretrained(model_path)
    model.vision_model.encoder.gradient_checkpointing = gradient_checkpointing
    model.vision_model.config.image_size = image_size
    model.interpolate_pos_embed(image_size)
    # NOTE we do not use pooler output
    model.vision_model.post_layernorm.requires_grad_(False)
    convert_clip_visual_attn(model)
    print(f"Freeze clip_vit_adapter_hf is {freeze}")
    model.requires_grad_((not freeze))
    print(f"Freeze vit of clip_vit_adapter_hf is {freeze_vit}")
    if freeze_vit:
        for name, param in model.vision_model.named_parameters():
            if not name.startswith("adapter"):
                param.requires_grad_(False)
    model.vision_model.set_vis_embed_requires_grad((not freeze))
    
    return model
