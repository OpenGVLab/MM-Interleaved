from typing import Optional, List, Union

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from transformers import LlamaConfig

from .encoders.visual_tokenizer import VisualTokenizer
from .decoders.modeling_llama_mmfs import (
    LlamaForCausalLM,
    LlamaModel,
)
from .decoders.decoder_text import TextDecoder
from .decoders.decoder_image import ImageDecoder
from .utils.causal_lm_cascade import CascadeLlamaForCausalLMWrapper
from .utils.pos_embed import get_1d_sincos_pos_embed_from_grid
from .utils.ops.modules import MMFS


class MMInterleaved(nn.Module):
    def __init__(
        self,
        *,
        llm_model_path="",
        seq_len=2048,
        txt_vocab_size=32002,
        loss_img_weight=10.0,
        loss_txt_weight=1.0,
        special_token_dict: dict = dict(
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=31999,
            soi_token_id=32000,
            image_token_id=32001,
        ),
        visual_tokenizer_config=None,
        image_decoder_config=None,
        use_llama_gradient_checkpointing=True,
        num_img_token=64,
        image_embed_dim=1024,
        cross_attention_frequency=4,
        spatial_shapes=[32, 16, 8],
        dataset_to_ignore_noimage_cond_loss=[],
    ):
        super().__init__()
        self.dataset_to_ignore_noimage_cond_loss = dataset_to_ignore_noimage_cond_loss

        self.seq_len = seq_len
        self.txt_vocab_size = txt_vocab_size
        self.special_token_dict = special_token_dict
        self.loss_img_weight = loss_img_weight
        self.loss_txt_weight = loss_txt_weight
        self.num_img_token = num_img_token

        llm_config = LlamaConfig.from_pretrained(llm_model_path)

        self.visual_tokenizer = VisualTokenizer(
            llm_hidden_size=llm_config.hidden_size,
            **visual_tokenizer_config
        )

        llm_config.image_embed_dim = image_embed_dim
        llm_config.cross_attention_frequency = cross_attention_frequency
        llm_config.spatial_shapes = spatial_shapes
        self.spatial_shapes = spatial_shapes
        llm_model = LlamaForCausalLM.from_pretrained(llm_model_path, config=llm_config)
        orig_txt_vocab_size = llm_model.config.vocab_size
        llm_model.resize_token_embeddings(txt_vocab_size)
        llm_model.requires_grad_(False)
        for k, v in llm_model.named_parameters():
            if "llama_cross_attn" in k:
                v.requires_grad = True
                print(f"set {k} requires_grad to True")
        self.mm_decoder: LlamaModel = llm_model.model  # LlamaModel
        self.mm_decoder.gradient_checkpointing = use_llama_gradient_checkpointing
        self.text_decoder = TextDecoder(
            config=llm_model.config,
            txt_vocab_size=txt_vocab_size,
            orig_txt_vocab_size=orig_txt_vocab_size,
        )
        self.text_decoder.init_from_llm(
            llm_model, orig_txt_vocab_size=orig_txt_vocab_size
        )

        hidden_size = self.mm_decoder.config.hidden_size
        if image_decoder_config is not None:
            self.image_decoder = ImageDecoder(
                **image_decoder_config,
                mmfs_input_channel=image_embed_dim,
            )
        else:
            self.image_decoder = None

        self.context_feat_proj = nn.Linear(hidden_size, hidden_size)
        self.soi_token = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)

        self.print_parameters_stats(prefix="MMInterleaved.")
        self.apply(self._init_mmfs_weights)

    def _init_mmfs_weights(self, m):
        if isinstance(m, MMFS):
            print("reinit weights of MMFS")
            m._reset_parameters()

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

    def _prepare_mm_embeds(
        self,
        text_ids: torch.LongTensor,
        image_tensors: Optional[torch.FloatTensor] = None,
        num_image_per_seq: Optional[torch.Tensor] = None,
        meta: Optional[Union[torch.Tensor, List]] = None,
    ):
        output = {}

        # step 1. get text token embeds
        text_embeds = self.mm_decoder.get_input_embeddings()(text_ids)
        B, L, C = text_embeds.shape

        assert num_image_per_seq.sum() == image_tensors.shape[0], (
            f"image_tensors.shape: {image_tensors.shape} | "
            f"num_image_per_seq.sum(): {num_image_per_seq.sum()}"
        )

        # step 2. get image embeds
        visual_output = self.visual_tokenizer(image_tensors)
        valid_image_embeds = visual_output["vis_embed"]
        valid_image_embeds = rearrange(valid_image_embeds, "b l c -> (b l) c")

        # step 3. insert image embeds into text embeds
        image_token_pos_x, image_token_pos_y = (
            text_ids == self.special_token_dict["image_token_id"]
        ).nonzero(as_tuple=True)
        image_token_pos = image_token_pos_x * L + image_token_pos_y
        assert image_token_pos.shape[0] == valid_image_embeds.shape[0], (
            f"{image_token_pos.shape=}, {valid_image_embeds.shape=}\n"
            f"{meta}\n"
            f"{text_ids[:,:100]} \n {text_ids[:,-100:]}"
        )
        text_embeds = rearrange(text_embeds, "b l c -> (b l) c")
        text_embeds = text_embeds.to(valid_image_embeds.dtype)
        image_token_pos = image_token_pos[:, None].expand(-1, C)
        mm_embeds = torch.scatter(
            text_embeds, dim=0, index=image_token_pos, src=valid_image_embeds
        )
        # add learnable soi token
        soi_token_pos_x, soi_token_pos_y = (
            text_ids == self.special_token_dict["soi_token_id"]
        ).nonzero(as_tuple=True)
        soi_token_pos = soi_token_pos_x * L + soi_token_pos_y
        soi_token_pos = soi_token_pos[:, None].expand(-1, C)
        learnable_soi_embeds = self.soi_token.repeat(soi_token_pos.shape[0], 1)
        mm_embeds = torch.scatter_add(
            mm_embeds, dim=0, index=soi_token_pos, src=learnable_soi_embeds
        )
        mm_embeds = rearrange(mm_embeds, "(b l) c -> b l c", b=B)
        output["mm_embeds"] = mm_embeds

        # step 4. prepare cross attention mask and MMFS features for mm decoder
        output.update(
            self._prepare_mmfs_features_for_mm_decoder(
                text_ids,
                num_image_per_seq,
                visual_output["multiscale_features"],
            )
        )
        output["multiscale_features"] = visual_output["multiscale_features"]

        return output

    def _prepare_mmfs_features_for_mm_decoder(
        self,
        text_ids: torch.LongTensor,
        num_image_per_seq: Optional[torch.Tensor] = None,
        multiscale_features=None,
    ):
        output = {}

        B, L = text_ids.shape

        max_num_image = num_image_per_seq.max()
        soi_token_pos = (text_ids == self.special_token_dict["soi_token_id"]).nonzero(
            as_tuple=True
        )[1]
        image_token_pos = -1 * torch.ones(B, max_num_image).type_as(soi_token_pos)
        start_idx = 0
        for i in range(B):
            image_token_pos[i, : num_image_per_seq[i]] = (
                soi_token_pos[start_idx : start_idx + num_image_per_seq[i]] + 1
            )
            start_idx = start_idx + num_image_per_seq[i]
        image_token_pos = image_token_pos[..., None].repeat(1, 1, L)  # [B, N, L]

        text_pos_idxs = torch.arange(L).type_as(text_ids)[None, :].repeat(B, 1)
        nearest_bos_ids = text_pos_idxs.masked_fill(
            text_ids != self.special_token_dict["bos_token_id"], -1
        )
        nearest_bos_ids = nearest_bos_ids.cummax(dim=1).values  # [B, L]
        index = torch.arange(L).to(text_ids.device)[None, None, :]
        attention_mask = (
            (image_token_pos > nearest_bos_ids[:, None, :])
            * (image_token_pos <= index)
            * (image_token_pos != -1)
        )  # [B, N, L]

        attention_mask = attention_mask.transpose(-1, -2).float()  # [B, L, N]
        output["cross_attention_mask"] = attention_mask

        mmfs_features = []
        for feat in multiscale_features:
            shape = int(feat.shape[-1])
            if shape in self.spatial_shapes:
                mmfs_features.append(feat)
        mmfs_features_new = [
            torch.zeros(
                B,
                max_num_image,
                *feat.shape[1:],
                device=feat.device,
                dtype=feat.dtype,
            )
            for feat in mmfs_features
        ]
        for feat, feat_n in zip(mmfs_features, mmfs_features_new):
            start_idx = 0
            for i in range(B):
                item = feat[start_idx : start_idx + num_image_per_seq[i]]
                feat_n[i, : item.shape[0], ...] = item
                start_idx = start_idx + num_image_per_seq[i]

        mmfs_features_mm = []
        for feat in mmfs_features_new:
            feat_n = rearrange(feat, "b n c h w -> b n (h w) c")
            mmfs_features_mm.append(feat_n)
        mmfs_features_mm = torch.cat(mmfs_features_mm, dim=2)
        output["mmfs_features_mm"] = mmfs_features_mm

        return output

    def _prepare_context_features_for_image_decoder(
        self,
        context_features: torch.Tensor,
        text_ids: torch.LongTensor,
        image_start_token_idx: Optional[torch.LongTensor] = None,
        nearest_bos_idxs: Optional[torch.LongTensor] = None,
    ):
        # directly extract context features for each image from original contexts
        if image_start_token_idx is None:
            # [B_I,]
            image_start_token_idx = (
                text_ids == self.special_token_dict["soi_token_id"]
            ).nonzero(as_tuple=True)[-1]
        assert len(image_start_token_idx) > 0

        if nearest_bos_idxs is None:
            nearest_bos_idxs = torch.zeros_like(image_start_token_idx)

        image_start_token_row_ids = (
            text_ids == self.special_token_dict["soi_token_id"]
        ).nonzero(as_tuple=True)[0]
        B_I = image_start_token_idx.shape[0]
        C = context_features.shape[-1]
        context_lengths = image_start_token_idx - nearest_bos_idxs + 1
        L_max = max(context_lengths)
        context_features_per_image = torch.zeros((B_I, L_max, C)).type_as(
            context_features
        )
        context_attention_mask_per_image = torch.zeros((B_I, L_max)).type_as(
            image_start_token_idx
        )
        for i in range(B_I):
            row_idx = image_start_token_row_ids[i]
            _context_features = context_features[
                row_idx, nearest_bos_idxs[i] : image_start_token_idx[i] + 1, :
            ]
            # add pos_embed to context
            _context_features = _context_features.flip(dims=(0,))
            context_features_per_image[i, : context_lengths[i], :] = _context_features
            context_attention_mask_per_image[i, : context_lengths[i]] = 1

        # add pos_embed to context
        pos_1d = np.arange(self.seq_len, dtype=np.float32)
        pos_embed_1d = get_1d_sincos_pos_embed_from_grid(C, pos_1d)
        pos_embed_1d = torch.from_numpy(pos_embed_1d).type_as(context_features)
        context_features_per_image = self.context_feat_proj(context_features_per_image)
        context_features_per_image = (
            context_features_per_image + pos_embed_1d[None, :L_max]
        )

        return context_features_per_image, context_attention_mask_per_image

    def _prepare_mmfs_features_for_image_decoder(
        self,
        multiscale_features: List[torch.Tensor],
        text_ids: torch.LongTensor,
        nearest_bos_idxs: Optional[torch.LongTensor] = None,
        num_image_per_seq: Optional[torch.Tensor] = None,
    ):
        L = text_ids.shape[1]
        B_I = num_image_per_seq.sum()
        assert B_I == multiscale_features[0].shape[0]

        image_start_token_idx_x, image_start_token_idx_y = (
            text_ids == self.special_token_dict["soi_token_id"]
        ).nonzero(as_tuple=True)
        image_start_token_idx = image_start_token_idx_x * L + image_start_token_idx_y

        if nearest_bos_idxs is None:
            nearest_bos_idxs = torch.zeros_like(image_start_token_idx)
        nearest_bos_idxs = image_start_token_idx_x * L + nearest_bos_idxs

        image_context_mask = nearest_bos_idxs[:, None] <= image_start_token_idx[None, :]
        image_context_mask = torch.tril(image_context_mask, diagonal=-1)  # [B_I, B_I]
        image_context_mask = torch.triu(image_context_mask, diagonal=-1)
        mmfs_features = [
            torch.zeros_like(feat)[:, None] for feat in multiscale_features
        ]
        mmfs_mask = torch.zeros((B_I, 1), dtype=torch.long, device=text_ids.device)

        for i in range(B_I):
            image_context_idxs = image_context_mask[i].nonzero(as_tuple=True)[-1]
            for ms_feat, mmfs_feat in zip(multiscale_features, mmfs_features):
                mmfs_feat[i, : len(image_context_idxs)] = ms_feat[image_context_idxs]
            mmfs_mask[i, : len(image_context_idxs)] = 1

        return mmfs_features, mmfs_mask

    def _prepare_gt_text_ids(
        self,
        text_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        ignore_prompt_token_offset=0,
        gt_text_ids: Optional[torch.LongTensor] = None,
        meta=None,
    ):
        if gt_text_ids is not None:
            return gt_text_ids[..., 1:]

        gt_text_ids = text_ids.clone()
        if isinstance(ignore_prompt_token_offset, int):
            gt_text_ids[:, :ignore_prompt_token_offset] = -100
        else:
            assert (
                len(ignore_prompt_token_offset) == gt_text_ids.shape[0]
            ), f"{len(ignore_prompt_token_offset)}, {gt_text_ids.shape}\n{ignore_prompt_token_offset}"
            for idx, offset in enumerate(ignore_prompt_token_offset):
                gt_text_ids[idx, :offset] = -100

        # ignore text loss without image as prefix
        ignore_noimage_cond_loss = (
            meta["dataset_name"] in self.dataset_to_ignore_noimage_cond_loss
        )
        if ignore_noimage_cond_loss:
            text_pos_idxs = (
                torch.arange(text_ids.shape[-1])
                .type_as(text_ids)[None, :]
                .repeat(text_ids.shape[0], 1)
            )
            nearest_bos_idxs = text_pos_idxs.masked_fill(
                text_ids != self.special_token_dict["bos_token_id"], -1
            )
            nearest_bos_idxs = nearest_bos_idxs.cummax(dim=1).values
            nearest_bos_idxs = torch.clamp(nearest_bos_idxs, min=0)

            nearest_soi_idxs = text_pos_idxs.masked_fill(
                text_ids != self.special_token_dict["soi_token_id"], -1
            )
            nearest_soi_idxs = nearest_soi_idxs.cummax(dim=1).values
            noimage_cond_token = torch.logical_or(
                nearest_soi_idxs < nearest_bos_idxs, nearest_soi_idxs == -1
            )
            gt_text_ids = gt_text_ids.masked_fill(noimage_cond_token, -100)

        gt_text_ids = gt_text_ids[:, 1:]
        gt_text_ids = gt_text_ids.masked_fill(
            text_ids[:, 1:] == self.special_token_dict["pad_token_id"], -100
        )
        gt_text_ids = gt_text_ids.masked_fill(
            text_ids[:, 1:] == self.special_token_dict["image_token_id"], -100
        )
        gt_text_ids = gt_text_ids.masked_fill(attention_mask[:, 1:] == 0, -100)

        is_bos_token = text_ids[:, :-1] == self.special_token_dict["bos_token_id"]
        is_soi_token = text_ids[:, 1:] == self.special_token_dict["soi_token_id"]
        is_bos2soi_token = torch.logical_and(is_bos_token, is_soi_token)
        gt_text_ids = gt_text_ids.masked_fill(is_bos2soi_token, -100)

        gt_text_ids = gt_text_ids.masked_fill(
            text_ids[:, 1:] == self.special_token_dict["bos_token_id"], -100
        )

        return gt_text_ids

    def forward(
        self,
        text_ids: torch.LongTensor,
        image_tensors: Optional[torch.FloatTensor] = None,
        image_tensors_dec: Optional[torch.FloatTensor] = None,
        num_image_per_seq: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        gt_text_ids: Optional[torch.LongTensor] = None,
        nearest_bos_idxs: Optional[torch.Tensor] = None,
        ignore_prompt_token_offset=0,
        loss_img_weight=None,
        loss_txt_weight=None,
        meta=None,
        image_loss_mask=None,
        **kwargs,
    ):
        output, loss = {}, 0.0

        _output = self._prepare_mm_embeds(
            text_ids=text_ids,
            image_tensors=image_tensors,
            num_image_per_seq=num_image_per_seq,
            meta=meta,
        )
        mm_embeds = _output.pop("mm_embeds")
        cross_attention_mask = _output.pop("cross_attention_mask", None)
        mmfs_features_mm = _output.pop("mmfs_features_mm", None)
        output.update(_output)

        # forward through the mm_decoder
        mm_embeds.requires_grad_(True)
        mm_outputs = self.mm_decoder(
            inputs_embeds=mm_embeds,
            attention_mask=attention_mask,
            vision_hidden_states=mmfs_features_mm,
            cross_attention_mask=cross_attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        mm_hidden_state = mm_outputs.last_hidden_state

        mm_hidden_state_txt = mm_hidden_state.clone()
        text_decode_outputs = self.text_decoder(
            inputs_embeds=mm_hidden_state_txt,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_logits = text_decode_outputs.logits
        text_logits = rearrange(text_logits, "b n c -> b c n")

        gt_text_ids = self._prepare_gt_text_ids(
            text_ids,
            attention_mask=attention_mask,
            ignore_prompt_token_offset=ignore_prompt_token_offset,
            gt_text_ids=gt_text_ids,
            meta=meta,
        )

        text_logits = text_logits.float()
        loss_txt = F.cross_entropy(
            text_logits[..., :-1].contiguous(),
            gt_text_ids.contiguous(),
            reduction="mean",
        )
        loss_txt_weight = (
            loss_txt_weight if loss_txt_weight is not None else self.loss_txt_weight
        )
        loss = loss + loss_txt * loss_txt_weight
        output["loss_txt"] = loss_txt.detach()

        # step 6. forward through the image_decoder
        if self.image_decoder is not None:
            mm_hidden_state_img = mm_hidden_state.clone()
            context_features = mm_hidden_state_img
            (
                context_features,
                context_attention_mask,
            ) = self._prepare_context_features_for_image_decoder(
                context_features,
                text_ids=text_ids,
                image_start_token_idx=None,
                nearest_bos_idxs=nearest_bos_idxs,
            )
            multiscale_features = output.pop("multiscale_features")
            (
                mmfs_features,
                mmfs_mask,
            ) = self._prepare_mmfs_features_for_image_decoder(
                multiscale_features,
                text_ids=text_ids,
                nearest_bos_idxs=nearest_bos_idxs,
                num_image_per_seq=num_image_per_seq,
            )
            loss_img = self.image_decoder(
                image_tensors=image_tensors
                if image_tensors_dec is None
                else image_tensors_dec,
                context_features=context_features,
                context_attention_mask=context_attention_mask,
                image_loss_mask=image_loss_mask,
                mmfs_features=mmfs_features,
                mmfs_mask=mmfs_mask,
            )
            loss_img_weight = (
                loss_img_weight if loss_img_weight is not None else self.loss_img_weight
            )
            loss = loss + loss_img.mean() * loss_img_weight
            output["loss_img"] = loss_img.mean().detach()

        output["loss"] = loss
        return output

    def generate_images(
        self,
        text_ids: torch.LongTensor,
        image_tensors: Optional[torch.FloatTensor] = None,
        num_image_per_seq: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        meta=None,
        target_image_idxs=None,
        **kwargs,
    ):
        output = {}

        _output = self._prepare_mm_embeds(
            text_ids=text_ids,
            image_tensors=image_tensors,
            num_image_per_seq=num_image_per_seq,
            meta=meta,
        )
        mm_embeds = _output.pop("mm_embeds")
        cross_attention_mask = _output.pop("cross_attention_mask")
        mmfs_features_mm = _output.pop("mmfs_features_mm")
        output.update(_output)

        # forward through the mm_decoder
        mm_outputs = self.mm_decoder(
            inputs_embeds=mm_embeds,
            attention_mask=attention_mask,
            vision_hidden_states=mmfs_features_mm,
            cross_attention_mask=cross_attention_mask,
            return_dict=True,
        )
        mm_hidden_state = mm_outputs.last_hidden_state

        multiscale_features = output.pop("multiscale_features")
        (
            mmfs_features,
            mmfs_mask,
        ) = self._prepare_mmfs_features_for_image_decoder(
            multiscale_features,
            text_ids=text_ids,
            nearest_bos_idxs=None,
            num_image_per_seq=num_image_per_seq,
        )

        context_features = mm_hidden_state
        (
            context_features,
            context_attention_mask,
        ) = self._prepare_context_features_for_image_decoder(
            context_features,
            text_ids=text_ids,
            image_start_token_idx=None,
        )

        if target_image_idxs is not None:
            context_features = torch.index_select(
                context_features, dim=0, index=target_image_idxs
            )
            context_attention_mask = torch.index_select(
                context_attention_mask, dim=0, index=target_image_idxs
            )
            mmfs_mask = torch.index_select(mmfs_mask, dim=0, index=target_image_idxs)
            mmfs_features = [
                torch.index_select(ms_feat, dim=0, index=target_image_idxs)
                for ms_feat in mmfs_features
            ]

        image_decoder_output = self.image_decoder.generate_images(
            context_features=context_features,
            context_attention_mask=context_attention_mask,
            mmfs_features=mmfs_features,
            mmfs_mask=mmfs_mask,
            **kwargs,
        )
        output.update(image_decoder_output)

        return output

    def generate_texts(
        self,
        text_ids: torch.LongTensor,
        image_tensors: Optional[torch.FloatTensor] = None,
        num_image_per_seq: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        meta=None,
        **kwargs,
    ):
        num_captions = kwargs.pop("num_captions", 1)

        # blip2 hyper-params
        max_length = kwargs.pop("max_length", 30)
        min_length = kwargs.pop("min_length", 8)
        num_beams = kwargs.pop("num_beams", 5)
        use_nucleus_sampling = kwargs.pop("use_nucleus_sampling", False)
        top_p = kwargs.pop("top_p", 0.9)
        repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        length_penalty = kwargs.pop("length_penalty", 1.0)
        temperature = kwargs.pop("temperature", 1)

        output = {}

        # step 1. prepare initial mm embeds
        _output = self._prepare_mm_embeds(
            text_ids=text_ids,
            image_tensors=image_tensors,
            num_image_per_seq=num_image_per_seq,
            meta=meta,
        )
        mm_embeds = _output.pop("mm_embeds")
        cross_attention_mask = _output.pop("cross_attention_mask")
        mmfs_features_mm = _output.pop("mmfs_features_mm")
        output.update(_output)

        # step 2. instantiate CausalLMWrapper
        llm_wrapper = CascadeLlamaForCausalLMWrapper(
            self.mm_decoder,
            self.text_decoder,
        )

        generate_text_ids = llm_wrapper.generate(
            input_ids=None,
            inputs_embeds=mm_embeds,
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            pad_token_id=self.special_token_dict["pad_token_id"],
            bos_token_id=self.special_token_dict["bos_token_id"],
            eos_token_id=[
                self.special_token_dict["eos_token_id"],
                self.special_token_dict["soi_token_id"],
            ],
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            vision_hidden_states=mmfs_features_mm,
            cross_attention_mask=cross_attention_mask,
        )

        output["text_ids"] = generate_text_ids

        return output

    def generate_scores(
        self,
        text_ids: List[torch.LongTensor],
        image_tensors: Optional[torch.FloatTensor] = None,
        num_image_per_seq: Optional[torch.Tensor] = None,
        attention_mask: Optional[List[torch.LongTensor]] = None,
        options_ids: Optional[List[torch.LongTensor]] = None,
        options_attn_masks: Optional[List[torch.LongTensor]] = None,
        **kwargs,
    ):
        output = {}

        scores = []
        for i in range(len(text_ids)):
            context_offset = len(text_ids[i])
            _text_ids = text_ids[i][None, ...].expand(options_ids[i].shape[0], -1)
            _text_ids = torch.cat((_text_ids, options_ids[i]), dim=1)
            _image_tensors = image_tensors[[i]].expand(
                options_ids[i].shape[0], -1, -1, -1
            )
            _num_image_per_seq = num_image_per_seq[[i]].expand(
                options_ids[i].shape[0], -1
            )
            _attention_mask = attention_mask[i][None, ...].expand(
                options_ids[i].shape[0], -1
            )
            _attention_mask = torch.cat((_attention_mask, options_attn_masks[i]), dim=1)

            mini_bs = 4
            mini_iter = math.ceil(options_ids[i].shape[0] / mini_bs)
            text_logits_all = []
            for j in range(mini_iter):
                # step 1. prepare initial mm embeds
                _output = self._prepare_mm_embeds(
                    text_ids=_text_ids[j * mini_bs : (j + 1) * mini_bs],
                    image_tensors=_image_tensors[j * mini_bs : (j + 1) * mini_bs],
                    num_image_per_seq=_num_image_per_seq[
                        j * mini_bs : (j + 1) * mini_bs
                    ],
                    meta=None,
                )
                mm_embeds = _output.pop("mm_embeds")
                cross_attention_mask = _output.pop("cross_attention_mask")
                mmfs_features_mm = _output.pop("mmfs_features_mm")

                # step 4. forward through the mm_decoder
                mm_outputs = self.mm_decoder(
                    inputs_embeds=mm_embeds,
                    attention_mask=_attention_mask[j * mini_bs : (j + 1) * mini_bs],
                    vision_hidden_states=mmfs_features_mm,
                    cross_attention_mask=cross_attention_mask,
                    return_dict=True,
                    output_hidden_states=True,
                )
                mm_hidden_state = mm_outputs.last_hidden_state
                text_decode_outputs = self.text_decoder(
                    inputs_embeds=mm_hidden_state,
                    attention_mask=_attention_mask[j * mini_bs : (j + 1) * mini_bs],
                    return_dict=True,
                )
                text_logits = text_decode_outputs.logits[:, context_offset - 1 : -1]
                text_logits_all.append(text_logits.detach())
            text_logits = torch.cat(text_logits_all)
            assert (
                text_logits.shape[1] == options_ids[i].shape[1]
            ), f"{text_logits.shape=} {options_ids[i].shape=}"

            text_log_probs = F.log_softmax(text_logits, dim=-1)
            text_log_probs = torch.gather(
                text_log_probs, dim=-1, index=options_ids[i][..., None]
            ).squeeze()
            text_scores = (text_log_probs * options_attn_masks[i]).sum(dim=-1)
            scores.append(text_scores.detach())

        scores = torch.stack(scores, dim=0)[:, None, :]
        output["scores"] = scores

        return output

    def generate(
        self,
        mode="generate_images",
        **kwargs,
    ):
        if mode == "generate_images":
            assert self.image_decoder is not None
            return self.generate_images(**kwargs)
        elif mode in ("generate_texts", "generate_vqa", "generate_grounding"):
            assert self.text_decoder is not None
            return self.generate_texts(**kwargs)
        elif mode == "generate_scores":
            assert self.text_decoder is not None
            return self.generate_scores(**kwargs)
        elif mode == "generate_segm":
            assert self.image_decoder is not None
            return self.generate_images(**kwargs)
        else:
            raise NotImplementedError
