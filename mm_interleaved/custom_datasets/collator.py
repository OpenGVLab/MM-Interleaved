import os
import random
from typing import Any
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel

from .wds_utils import init_tokenizer
from .collator_sft import MultiImageCollator

def build_data_collator(config, train_dataset=None):
    collator_name = getattr(config, "collator", "")
    if not collator_name:
        return None
    if collator_name == "ImageTextPairCollator":
        return ImageTextPairCollator(
            tokenizer_path=config.tokenizer_path,
            mode=getattr(config, "collate_mode", "train"),
            uncond_prob=getattr(config, "uncond_prob", 0.0),
            num_img_token=getattr(config, "num_img_token", 32),
            img_first_prob=getattr(config, "img_first_prob", 1.0),
            text_prompt=getattr(config, "text_prompt", "a photo of "),
            add_soi_token=getattr(config, "add_soi_token", True),
            generation_kwargs=getattr(config, "generation_kwargs", None),
            use_instr_format=getattr(config, "use_instr_format", True),
            instr_prompts=getattr(config, "instr_prompts", None),
            padding=getattr(config, "padding", "longest"),
            train_dataset=train_dataset,
            few_shot_n_shot=getattr(config, "few_show_n_shot", 2),
            few_shot_template=getattr(
                config,
                "few_shot_template",
                "Caption: {caption}",
            ),
            use_rice=getattr(config, "use_rice", False),
            rice_encoder=getattr(
                config, "rice_encoder", "./assets/openai/clip-vit-large-patch14"
            ),
            cached_features_path=getattr(
                config, "cached_features_path", "./OUTPUT/cached_feature"
            ),
        )
    elif collator_name == "VQACollator":
        return VQACollator(
            tokenizer_path=config.tokenizer_path,
            mode=getattr(config, "collate_mode", "train"),
            num_img_token=getattr(config, "num_img_token", 32),
            text_prompt=getattr(
                config,
                "text_prompt",
                "Based on the image, please answer the question. {image}{question} Please provide an accurate answer within one word. The answer is:",
            ),
            add_soi_token=getattr(config, "add_soi_token", True),
            generation_kwargs=getattr(config, "generation_kwargs", None),
            use_instr_format=getattr(config, "use_instr_format", True),
            instr_prompts=getattr(config, "instr_prompts", None),
            train_dataset=train_dataset,
            few_shot_n_shot=getattr(config, "few_shot_n_shot", 2),
            few_shot_template=getattr(
                config,
                "few_shot_template",
                "Question: {question} Short answer: {answer}{eos_token}",
            ),
            use_rice=getattr(config, "use_rice", False),
            rice_encoder=getattr(
                config, "rice_encoder", "./assets/openai/clip-vit-large-patch14"
            ),
            cached_features_path=getattr(
                config, "cached_features_path", "./OUTPUT/cached_feature"
            ),
        )
    elif collator_name == "MultiImageCollator":
        return MultiImageCollator(
            tokenizer_path=config.tokenizer_path,
            mode=getattr(config, "collate_mode", "train"),
            generation_kwargs=getattr(config, "generation_kwargs", None),
            padding=getattr(config, "padding", "longest"),
            ignore_image_loss_idx=getattr(config, "ignore_image_loss_idx", -1),
        )
    elif collator_name == "GroundingCollator":
        return GroundingCollator(
            tokenizer_path=config.tokenizer_path,
            mode=getattr(config, "collate_mode", "train"),
            task=getattr(config, "collate_task", "grounding"),
            num_img_token=getattr(config, "num_img_token", 32),
            text_prompt=getattr(config, "text_prompt", None),
            add_soi_token=getattr(config, "add_soi_token", True),
            generation_kwargs=getattr(config, "generation_kwargs", None),
            use_instr_format=getattr(config, "use_instr_format", False),
            instr_prompts=getattr(config, "instr_prompts", None),
            ignore_soi_token_loss=getattr(config, "ignore_soi_token_loss", None),
            ignore_bos2soi_token_loss=getattr(config, "ignore_bos2soi_token_loss", None),
            max_length=getattr(config, "max_length", 2048),
            force_3_digits=getattr(config, "force_3_digits", True),
        )
    elif collator_name == "VisDialCollator":
        return VisDialCollator()

    return None


def interleaved_collation_fn(samples, pad_token_id=-1, return_nearest_bos_idxs=False, loss_img_weight=None, loss_txt_weight=None):
    image_tensors_all = []
    image_tensors_dec_all = []
    text_ids_all = []
    text_attn_mask_all = []
    num_images_all = []
    if return_nearest_bos_idxs:
        nearest_bos_idxs_all = []
    metas = []

    for sample in samples:
        image_tensors, text_ids, text_attn_mask = (
            sample["image_tensors"],
            sample["text_ids"],
            sample["text_attn_mask"],
        )
        image_tensors_all.append(torch.from_numpy(image_tensors))
        text_ids_all.append(torch.from_numpy(text_ids))
        text_attn_mask_all.append(torch.from_numpy(text_attn_mask))
        num_images_all.append(image_tensors.shape[0])
        if return_nearest_bos_idxs:
            nearest_bos_idxs_all.append(torch.from_numpy(sample["nearest_bos_idxs"]))
        if sample.get("image_tensors_dec", None) is not None:
            image_tensors_dec_all.append(torch.from_numpy(sample["image_tensors_dec"]))
        if "meta" in sample:
            metas.append(sample["meta"])

    # pad text_ids
    seq_lens = [len(text_id) for text_id in text_ids_all]
    metas = dict(meta=metas, seq_lens=torch.tensor(seq_lens))
    if pad_token_id > 0 and len(set(seq_lens)) > 1:
        text_ids = torch.nn.utils.rnn.pad_sequence(
            text_ids_all, batch_first=True, padding_value=pad_token_id
        )
        text_attn_mask = torch.nn.utils.rnn.pad_sequence(
            text_attn_mask_all, batch_first=True, padding_value=0
        )
    else:
        text_ids = torch.stack(text_ids_all, dim=0)
        text_attn_mask = torch.stack(text_attn_mask_all, dim=0)

    image_tensors = torch.cat(image_tensors_all)
    num_images = torch.tensor(num_images_all)
    if return_nearest_bos_idxs:
        nearest_bos_idxs = torch.cat(nearest_bos_idxs_all)
    if len(image_tensors_dec_all) > 0:
        image_tensors_dec = torch.cat(image_tensors_dec_all)
        assert image_tensors_dec.shape[0] == image_tensors.shape[0]
    else:
        image_tensors_dec = None

    data = dict(
        image_tensors=image_tensors,
        image_tensors_dec=image_tensors_dec,
        text_ids=text_ids,
        attention_mask=text_attn_mask,
        num_image_per_seq=num_images,
        nearest_bos_idxs=nearest_bos_idxs if return_nearest_bos_idxs else None,
        meta=metas,
        loss_img_weight=loss_img_weight,
        loss_txt_weight=loss_txt_weight,
    )

    return data

class ImageTextPairCollator:
    def __init__(
        self,
        tokenizer_path,
        mode="train",
        uncond_prob=0.0,
        num_img_token=32,
        img_first_prob=1.0,
        text_prompt="a photo of ",
        add_soi_token=True,
        generation_kwargs=None,
        use_instr_format=True,
        instr_prompts=None,
        padding="longest",
        train_dataset=None,
        few_shot_n_shot=2,
        few_shot_template="Caption: {caption}",
        use_rice=False,
        rice_encoder="./assets/openai/clip-vit-large-patch14",
        cached_features_path=None,
    ):
        self.tokenizer = init_tokenizer(tokenizer_path)
        self.mode = mode
        self.num_img_token = num_img_token
        self.img_first_prob = img_first_prob
        self.text_prompt = text_prompt
        self.uncond_prob = uncond_prob
        self.add_soi_token = add_soi_token
        default_generation_kwargs = dict(
            max_length=20,
            min_length=8,
            length_penalty=1.,
            num_beams=5,
            top_p=0.9,
        )
        self.generation_kwargs = generation_kwargs or default_generation_kwargs
        self.padding = padding

        self.use_instr_format = use_instr_format
        default_instr_prompts = {
            "image": ["", "", ""],
            "text": [
                "a photo of",
                "{image}",
                "",
            ],
        }
        self.instr_prompts = instr_prompts or default_instr_prompts

        self.use_rice = use_rice
        self.train_dataset = train_dataset
        self.few_shot_n_shot = few_shot_n_shot
        self.few_shot_template = few_shot_template

        if self.use_rice:
            self.rice = RICES(
                dataset=self.train_dataset,
                batch_size=32,
                vision_encoder_path=rice_encoder,
                cached_features_path=cached_features_path,
            )

        self.image_subseq = "<|image|>" * self.num_img_token
        if self.add_soi_token:
            self.image_subseq = "<|beginofimage|>" + self.image_subseq

        self.echo = True
        print(
            "caption prompt template:",
            self.instr_prompts if self.use_instr_format else self.text_prompt,
        )

    def set_mode(self, mode):
        self.mode = mode

    def __call__(self, data_list) -> Any:
        if self.mode == "train":
            return self._call_for_train(data_list)
        elif self.mode == "generate_texts":
            return self._call_for_generate_texts(data_list)
        elif self.mode == "generate_images":
            return self._call_for_generate_images(data_list)
        elif self.mode == "generate_both":
            raise NotImplementedError(
                f"Get {self.mode}, please specify the exact mode before calling it"
            )
        else:
            raise NotImplementedError(
                f"collate_mode {self.mode} is NOT supported by far"
            )

    def _call_for_generate_texts(self, data_list, is_train=False):
        images_tensors_all = []
        num_image_per_seq = []
        image_tensors_dec_all = []
        meta = []

        text_inputs_with_prompt_image_all = []

        if self.use_instr_format:
            assis_prompt, user_prompt, sys_prompt = self.instr_prompts["text"]
        else:
            assis_prompt, user_prompt, sys_prompt = "", self.text_prompt, ""
        if "{image}" not in user_prompt:
            user_prompt = "{image}" + user_prompt

        use_few_shot = (
            "{few_shot_example}" in user_prompt and self.train_dataset is not None
        )

        # ignore text_prompt token when calculating loss during training
        ignore_prompt_token_offsets = []

        for data in data_list:
            images_tensor, caption, index = data
            meta.append((index, caption))

            if isinstance(images_tensor, np.ndarray):
                images_tensor = torch.from_numpy(images_tensor)
                _images_tensor_all = [images_tensor]
                _image_tensors_dec_all = []
            elif isinstance(images_tensor, tuple):
                images_tensor, images_tensor_dec = images_tensor
                images_tensor = torch.from_numpy(images_tensor)
                _images_tensor_all = [images_tensor]
                images_tensor_dec = torch.from_numpy(images_tensor_dec)
                _image_tensors_dec_all = [images_tensor_dec]

            _num_image_per_seq = 1

            if use_few_shot:
                few_shot_example, images = self.get_few_shot_samples(
                    query_image=images_tensor
                )
                text_input = user_prompt.format(
                    few_shot_example=few_shot_example,
                    image=self.image_subseq,
                )
                # few-shot images first, then question image
                if isinstance(images, tuple):
                    _images_tensor_all = images[0] + _images_tensor_all
                    _image_tensors_dec_all = images[1] + _image_tensors_dec_all
                else:
                    _images_tensor_all = images + _images_tensor_all
                _num_image_per_seq += len(images)
            else:
                text_input = user_prompt.format(image=self.image_subseq)

            text_input = f"{sys_prompt} {text_input} {assis_prompt}".strip()

            images_tensors_all.extend(_images_tensor_all)
            image_tensors_dec_all.extend(_image_tensors_dec_all)
            num_image_per_seq.append(_num_image_per_seq)

            if is_train:
                ignore_prompt_token_offset = self.tokenizer(
                    text_input.strip(), return_tensors="pt"
                ).attention_mask.sum(1)
                ignore_prompt_token_offsets.append(ignore_prompt_token_offset)
                text_input += " " + caption
            text_inputs_with_prompt_image_all.append(text_input)

            if self.echo:
                self.echo = False
                print("caption prompt example:", text_input)

        self.tokenizer.padding_side = "right" if is_train else "left"
        text_tensor = self.tokenizer(
            text_inputs_with_prompt_image_all,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_ids = text_tensor.input_ids
        attn_mask = text_tensor.attention_mask

        images_tensors = torch.stack(images_tensors_all, dim=0)
        image_tensors_dec = None
        if len(image_tensors_dec_all) > 0:
            image_tensors_dec = torch.stack(image_tensors_dec_all, dim=0)
            assert image_tensors_dec.shape[0] == images_tensors.shape[0]

        num_image_per_seq = torch.tensor(
            num_image_per_seq, dtype=torch.long, device=images_tensors.device
        )

        data = dict(
            image_tensors=images_tensors,
            image_tensors_dec=image_tensors_dec,
            num_image_per_seq=num_image_per_seq,
            text_ids=text_ids,
            attention_mask=attn_mask,
            loss_img_weight=0.0,
            ignore_prompt_token_offset=ignore_prompt_token_offsets,
            meta=meta,
        )

        if self.generation_kwargs is not None:
            for k, v in self.generation_kwargs.items():
                data[k] = v

        return data

    def _call_for_generate_images(self, data_list, is_train=False):
        images_tensors_all = []
        image_tensors_dec_all = []
        captions = []
        meta = []

        image_subseq = "<|image|>" * self.num_img_token
        if self.add_soi_token:
            image_subseq = "<|beginofimage|>" + image_subseq

        for data in data_list:
            images_tensor, caption, index = data
            if isinstance(images_tensor, np.ndarray):
                images_tensor = torch.from_numpy(images_tensor)
            elif isinstance(images_tensor, tuple):
                images_tensor, images_tensor_dec = images_tensor
                images_tensor = torch.from_numpy(images_tensor)
                images_tensor_dec = torch.from_numpy(images_tensor_dec)
                image_tensors_dec_all.append(images_tensor_dec)
            images_tensors_all.append(images_tensor)

            text = "" if is_train and np.random.random() < self.uncond_prob else caption

            meta.append((index, text))
            if self.use_instr_format:
                assis_prompt, user_prompt, sys_prompt = self.instr_prompts["image"]
                text = (
                    f"{sys_prompt} {user_prompt} {text} {assis_prompt} {image_subseq}"
                ).strip()
                text = text.replace("<|image|> ", "<|image|>").replace(
                    " <|beginofimage|>", "<|beginofimage|>"
                )
            else:
                text = text + image_subseq
            captions.append(text)

        images_tensors = torch.stack(images_tensors_all, dim=0)
        if len(image_tensors_dec_all) > 0:
            image_tensors_dec = torch.stack(image_tensors_dec_all, dim=0)
            assert image_tensors_dec.shape[0] == images_tensors.shape[0]
        else:
            image_tensors_dec = None

        self.tokenizer.padding_side = "right"
        text_tensor = self.tokenizer(
            captions,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding=self.padding,
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_ids = text_tensor.input_ids
        attn_mask = text_tensor.attention_mask
        num_image_per_seq = torch.ones(
            (images_tensors.shape[0],), dtype=torch.long, device=images_tensors.device
        )

        # prepare negative prompt_ids only for inference
        negative_prompt_ids = None
        if not is_train and self.uncond_prob > 0.0:
            negative_prompt = image_subseq
            negative_prompt_tensor = self.tokenizer(
                [negative_prompt],
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                padding=self.padding,
                return_tensors="pt",
                return_attention_mask=True,
            )
            negative_prompt_ids = negative_prompt_tensor.input_ids

        data = dict(
            image_tensors=images_tensors,
            image_tensors_dec=image_tensors_dec,
            num_image_per_seq=num_image_per_seq,
            text_ids=text_ids,
            attention_mask=attn_mask,
            negative_prompt_ids=negative_prompt_ids,
            loss_txt_weight=0.0,
            meta=meta,
        )

        if self.generation_kwargs is not None:
            for k, v in self.generation_kwargs.items():
                data[k] = v

        return data

    def _call_for_train(self, data_list):
        if np.random.random() < self.img_first_prob:
            # image to text
            return self._call_for_generate_texts(data_list, is_train=True)
        else:
            # text to image
            return self._call_for_generate_images(data_list, is_train=True)

    def get_few_shot_samples(self, query_image=None):
        images, images_dec = [], []

        if self.use_rice:
            samples = self.rice.find(query_image, self.few_shot_n_shot)[0]
        else:
            idx = random.sample(
                list(range(len(self.train_dataset))), self.few_shot_n_shot
            )
            samples = [self.train_dataset[i] for i in idx]

        few_shot_caption_only = "{image}" not in self.few_shot_template

        few_shot_example = ""
        for image, caption, _ in samples:
            if few_shot_caption_only:
                few_shot_example += self.few_shot_template.format(
                    caption=caption,
                )
            else:
                if isinstance(image, tuple):
                    images.append(
                        torch.from_numpy(image[0])
                        if isinstance(image[0], np.ndarray)
                        else image[0]
                    )
                    images_dec.append(
                        torch.from_numpy(image[1])
                        if isinstance(image[1], np.ndarray)
                        else image[1]
                    )
                else:
                    images.append(
                        torch.from_numpy(image)
                        if isinstance(image, np.ndarray)
                        else image
                    )
                few_shot_example += self.few_shot_template.format(
                    image=self.image_subseq,
                    caption=caption,
                )

        images = (images, images_dec) if len(images_dec) > 0 else images

        return few_shot_example, images


class VQACollator:
    def __init__(
        self,
        tokenizer_path,
        mode="train",
        num_img_token=32,
        text_prompt="Based on the image, please answer the question. {image}{question} Please provide an accurate answer within one word. The answer is:",
        add_soi_token=True,
        generation_kwargs=None,
        use_instr_format=True,
        instr_prompts=None,
        train_dataset=None,
        few_shot_n_shot=2,
        few_shot_template="Question: {question} Short answer: {answer}{eos_token}",
        use_rice=False,
        rice_encoder="./assets/openai/clip-vit-large-patch14",
        cached_features_path=None,
    ):
        self.tokenizer = init_tokenizer(tokenizer_path)
        self.mode = mode
        self.num_img_token = num_img_token
        self.text_prompt = text_prompt

        self.add_soi_token = add_soi_token
        default_generation_kwargs = dict(
            max_length=10,
            min_length=0,
            length_penalty=0.,
            num_beams=3,
            top_p=1.0,
        )
        self.generation_kwargs = generation_kwargs or default_generation_kwargs

        self.use_instr_format = use_instr_format
        default_instr_prompts = [
            "The answer is:",
            "Based on the image, please answer the question. {image}{question} Please provide an accurate answer within one word.",
            "",
        ]
        self.instr_prompts = instr_prompts or default_instr_prompts

        self.use_rice = use_rice
        self.train_dataset = train_dataset
        self.few_shot_n_shot = few_shot_n_shot
        self.few_shot_template = few_shot_template

        if self.use_rice:
            self.rice = RICES(
                dataset=self.train_dataset,
                batch_size=32,
                vision_encoder_path=rice_encoder,
                cached_features_path=cached_features_path,
            )

        self.image_subseq = "<|image|>" * self.num_img_token
        if self.add_soi_token:
            self.image_subseq = "<|beginofimage|>" + self.image_subseq

        self.echo = True
        print(
            "vqa prompt template:",
            self.instr_prompts if self.use_instr_format else self.text_prompt,
        )

    def set_mode(self, mode):
        self.mode = mode

    def __call__(self, data_list) -> Any:
        return self._call_for_generate_texts(data_list, is_train=self.mode == "train")

    def _call_for_generate_texts(self, data_list, is_train=False):
        meta = []
        images_tensors_all = []
        num_image_per_seq = []
        text_inputs_with_prompt_image_all = []

        if self.use_instr_format:
            assis_prompt, user_prompt, sys_prompt = self.instr_prompts
        else:
            assis_prompt, user_prompt, sys_prompt = "", self.text_prompt, ""
        assert "{image}" in user_prompt and "{question}" in user_prompt

        use_few_shot = (
            "{few_shot_example}" in user_prompt and self.train_dataset is not None
        )

        # ignore text_prompt token when calculating loss during training
        ignore_prompt_token_offsets = []

        for data in data_list:
            images_tensor, question, answer, index = data
            if isinstance(images_tensor, np.ndarray):
                images_tensor = torch.from_numpy(images_tensor)
            meta.append((index, question, answer))

            _images_tensor_all = [images_tensor]
            _num_image_per_seq = 1

            if use_few_shot:
                few_shot_example, images = self.get_few_shot_samples(
                    query_image=images_tensor
                )
                text_input = user_prompt.format(
                    few_shot_example=few_shot_example,
                    image=self.image_subseq,
                    question=question,
                )
                # few-shot images first, then question image
                _images_tensor_all = images + _images_tensor_all
                _num_image_per_seq += len(images)
            else:
                text_input = user_prompt.format(
                    image=self.image_subseq, question=question
                )

            text_input = f"{sys_prompt} {text_input} {assis_prompt}".strip()
            images_tensors_all.extend(_images_tensor_all)
            num_image_per_seq.append(_num_image_per_seq)
            if is_train:
                ignore_prompt_token_offset = self.tokenizer(
                    text_input.strip(), return_tensors="pt"
                ).attention_mask.sum(1)
                ignore_prompt_token_offsets.append(ignore_prompt_token_offset)
                text_input += " " + answer + self.tokenizer.eos_token
            text_inputs_with_prompt_image_all.append(text_input)

            if self.echo:
                self.echo = False
                print("vqa prompt example:", text_input)

        self.tokenizer.padding_side = "right" if is_train else "left"
        text_tensor = self.tokenizer(
            text_inputs_with_prompt_image_all,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_ids = text_tensor.input_ids
        attn_mask = text_tensor.attention_mask

        images_tensors = torch.stack(images_tensors_all, dim=0)
        num_image_per_seq = torch.tensor(
            num_image_per_seq, dtype=torch.long, device=images_tensors.device
        )

        data = dict(
            image_tensors=images_tensors,
            num_image_per_seq=num_image_per_seq,
            text_ids=text_ids,
            attention_mask=attn_mask,
            loss_img_weight=0.0,
            ignore_prompt_token_offset=ignore_prompt_token_offsets,
            meta=meta,
        )

        if self.generation_kwargs is not None:
            for k, v in self.generation_kwargs.items():
                data[k] = v

        return data

    def get_few_shot_samples(self, query_image=None):
        images = []

        if self.use_rice:
            samples = self.rice.find(query_image, self.few_shot_n_shot)[0]
        else:
            idx = random.sample(
                list(range(len(self.train_dataset))), self.few_shot_n_shot
            )
            samples = [self.train_dataset[i] for i in idx]

        few_shot_caption_only = "{image}" not in self.few_shot_template
        few_shot_image_only = "{question}" not in self.few_shot_template

        few_shot_example = ""
        for image, question, answer, _ in samples:
            if few_shot_caption_only:
                few_shot_example += self.few_shot_template.format(
                    question=question,
                    answer=answer,
                    eos_token="",
                )
            elif few_shot_image_only:
                images.append(
                    torch.from_numpy(image) if isinstance(image, np.ndarray) else image
                )
                few_shot_example += self.few_shot_template.format(
                    image=self.image_subseq,
                )
            else:
                images.append(
                    torch.from_numpy(image) if isinstance(image, np.ndarray) else image
                )
                few_shot_example += self.few_shot_template.format(
                    image=self.image_subseq,
                    question=question,
                    answer=answer,
                    eos_token="",
                )

        return few_shot_example, images


class GroundingCollator:
    def __init__(
        self,
        tokenizer_path,
        mode="train",
        task="grounding",
        num_img_token=32,
        text_prompt=None,
        add_soi_token=True,
        generation_kwargs=None,
        use_instr_format=False,
        instr_prompts=None,
        ignore_soi_token_loss=False,
        ignore_bos2soi_token_loss=False,
        max_length=2048,
        force_3_digits=True,
    ):
        assert task in ("grounding", "referring", "region_vqa", "grounded_caption")
        
        self.tokenizer = init_tokenizer(tokenizer_path)
        self.mode = mode
        self.collate_mode = mode
        self.task = task
        self.num_img_token = num_img_token
        self.max_length = max_length
        self.force_3_digits = force_3_digits

        self.ignore_soi_token_loss = ignore_soi_token_loss
        self.ignore_bos2soi_token_loss = ignore_bos2soi_token_loss

        self.add_soi_token = add_soi_token
        self.generation_kwargs = generation_kwargs

        self.use_instr_format = use_instr_format
        
        if task == "grounding":
            default_instr_prompts = [
                "ASSISTANT:",
                "USER: {image}Provide the bounding box coordinate of the region this sentence describes. {caption}",
                "You are a helpful assistant.",
            ]
            default_text_prompt = "{image}Provide the bounding box coordinate of the region this sentence describes. {caption}"
        elif task == "referring":
            default_instr_prompts = [
                "ASSISTANT:",
                "USER: {image}Provide a short description for this <ref>region</ref><box>{box}</box>.",
                "You are a helpful assistant.",
            ]
            default_text_prompt = "{image}Provide a short description for this <ref>region</ref><box>{box}</box>."
        elif task == "region_vqa":
            default_instr_prompts = [
                "ASSISTANT:",
                "USER: {image}Answer this question according to the <ref>region</ref><box>{box}</box>. {question}",
                "You are a helpful assistant.",
            ]
            default_text_prompt = "{image}Answer this question according to the <ref>region</ref><box>{box}</box>. {question}"
        elif task == "grounded_caption":
            default_instr_prompts = [
                "ASSISTANT:",
                "USER: {image}Generate the caption with grounding.",
                "You are a helpful assistant.",
            ]
            default_text_prompt = "{image}Generate the caption with grounding."
        else:
            raise NotImplementedError
            
            
        self.text_prompt = text_prompt or default_text_prompt
        self.instr_prompts = instr_prompts or default_instr_prompts

        self.image_subseq = "<|image|>" * self.num_img_token
        if self.add_soi_token:
            self.image_subseq = "<|beginofimage|>" + self.image_subseq

        self.echo = True
        print(
            "vqa prompt template:",
            self.instr_prompts if self.use_instr_format else self.text_prompt,
        )

    def set_mode(self, mode):
        self.mode = mode
        self.collate_mode = mode

    def box2str(self, box):
        x1, y1, x2, y2 = box
        assert x1 <= x2 and y1 <= y2

        if self.force_3_digits:
            return f"({x1:03d},{y1:03d})({x2:03d},{y2:03d})"
        else:
            return f"({x1},{y1})({x2},{y2})"

    def __call__(self, data_list) -> Any:
        concat_mode = [data.get('concat_mode', False) for data in data_list]
        assert all(concat_mode) or not any(concat_mode)
        
        if all(concat_mode):
            return self._call_for_concat_mode(data_list)
        
        return self._call_for_generate_texts(data_list, is_train=self.mode == "train")

    def _call_for_generate_texts(self, data_list, is_train=False):
        meta = []
        images_tensors_all = []
        num_image_per_seq = []
        text_inputs_with_prompt_image_all = []

        if self.use_instr_format:
            assis_prompt, user_prompt, sys_prompt = self.instr_prompts
        else:
            assis_prompt, user_prompt, sys_prompt = "", self.text_prompt, ""

        # ignore text_prompt token when calculating loss during training
        ignore_prompt_token_offsets = []

        for data in data_list:
            images_tensor = data['images_tensor']
            question = data.get('query', None)  # None if self.task is not "region_vqa"
            answer = data['label']
            index = data['id']
            
            if isinstance(images_tensor, np.ndarray):
                images_tensor = torch.from_numpy(images_tensor)
            meta.append((index, question, answer, data['image'].height, data['image'].width, data.get('bbox', None)))

            _images_tensor_all = [images_tensor]
            _num_image_per_seq = 1

            if self.task == 'grounding':
                box = self.box2str(data['bbox'])
                text_input = user_prompt.format(
                    image=self.image_subseq, caption=answer,
                ) + '<box>'
            elif self.task == "referring":
                box = self.box2str(data['bbox'])
                text_input = user_prompt.format(
                    image=self.image_subseq, box=box,
                )
            elif self.task == 'region_vqa':
                box = self.box2str(data['bbox'])
                text_input = user_prompt.format(
                    image=self.image_subseq, question=question, box=box,
                )
            else:
                text_input = user_prompt.format(image=self.image_subseq)

            text_input = f"{sys_prompt} {text_input} {assis_prompt}".strip()
            images_tensors_all.extend(_images_tensor_all)
            num_image_per_seq.append(_num_image_per_seq)

            if is_train:
                ignore_prompt_token_offset = self.tokenizer(
                    text_input.strip(), return_tensors="pt"
                ).attention_mask.sum(1)
                ignore_prompt_token_offsets.append(ignore_prompt_token_offset)
                
                if self.task == "grounding":
                    box = self.box2str(data['bbox'])
                    # text_input += f" <box>{box}</box>{self.tokenizer.eos_token}"
                    text_input += f"{box}</box>{self.tokenizer.eos_token}"
                else:
                    text_input += " " + answer + self.tokenizer.eos_token
            text_inputs_with_prompt_image_all.append(text_input)

            if self.echo:
                self.echo = False
                print("vqa prompt example:", text_input)

        self.tokenizer.padding_side = "right" if is_train else "left"
        text_tensor = self.tokenizer(
            text_inputs_with_prompt_image_all,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
            max_length=self.max_length,
        )
        text_ids = text_tensor.input_ids
        attn_mask = text_tensor.attention_mask

        images_tensors = torch.stack(images_tensors_all, dim=0)
        num_image_per_seq = torch.tensor(
            num_image_per_seq, dtype=torch.long, device=images_tensors.device
        )

        if is_train:
            # Modified from _prepare_gt_text_ids()
            gt_text_ids = text_ids.clone()
            assert gt_text_ids.shape[0] == len(ignore_prompt_token_offsets), f'{gt_text_ids.shape[0]},{len(ignore_prompt_token_offsets)}'
            for idx, offset in enumerate(ignore_prompt_token_offsets):
                gt_text_ids[idx, :offset] = -100

            gt_text_ids = gt_text_ids.masked_fill(
                text_ids == self.tokenizer.pad_token_id, -100
            )
            gt_text_ids = gt_text_ids.masked_fill(
                text_ids == self.tokenizer.convert_tokens_to_ids('<|image|>'), -100
            )
            gt_text_ids = gt_text_ids.masked_fill(attn_mask == 0, -100)
            if self.ignore_bos2soi_token_loss:
                is_bos_token = text_ids[:-1] == self.tokenizer.convert_tokens_to_ids('<s>')
                is_soi_token = text_ids[1:] == self.tokenizer.convert_tokens_to_ids('<|beginofimage|>')
                is_bos2soi_token = torch.logical_and(is_bos_token, is_soi_token)
                gt_text_ids[1:] = gt_text_ids[1:].masked_fill(is_bos2soi_token, -100)
            if self.ignore_soi_token_loss:
                gt_text_ids = gt_text_ids.masked_fill(
                    text_ids == self.tokenizer.convert_tokens_to_ids('<|beginofimage|>'), -100
                )
            gt_text_ids = gt_text_ids.contiguous()
        else:
            gt_text_ids = None

        data = dict(
            image_tensors=images_tensors,
            num_image_per_seq=num_image_per_seq,
            text_ids=text_ids,
            attention_mask=attn_mask,
            gt_text_ids=gt_text_ids,
            loss_img_weight=0.0,
            ignore_prompt_token_offset=ignore_prompt_token_offsets,
            meta=meta,
        )

        if self.generation_kwargs is not None:
            for k, v in self.generation_kwargs.items():
                data[k] = v

        return data

    def _call_for_concat_mode(self, data_list):
        image_tensors = []
        num_image_per_seq = []
        text_ids = []
        attn_mask = []
        gt_text_ids = []

        for data in data_list:
            image_tensors.append(data['image_tensors'])
            num_image_per_seq.append(data['num_image_per_seq'])
            
            assert data['text_ids'].shape[0] == 1
            assert data['attention_mask'].shape[0] == 1
            assert data['gt_text_ids'].shape[0] == 1
            
            text_ids.append(data['text_ids'].squeeze(0))
            attn_mask.append(data['attention_mask'].squeeze(0))
            gt_text_ids.append(data['gt_text_ids'].squeeze(0))

        image_tensors = torch.cat(image_tensors)
        num_image_per_seq = torch.stack(num_image_per_seq)
        text_ids = torch.nn.utils.rnn.pad_sequence(text_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn_mask = torch.nn.utils.rnn.pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gt_text_ids = torch.nn.utils.rnn.pad_sequence(gt_text_ids, batch_first=True, padding_value=-100)

        data = dict(
            image_tensors=image_tensors,
            num_image_per_seq=num_image_per_seq,
            text_ids=text_ids,
            attention_mask=attn_mask,
            gt_text_ids=gt_text_ids,
            loss_img_weight=0.0,
        )

        return data


class VisDialCollator:
    def __init__(self):
        pass

    def __call__(self, data_list):
        image_ids = []
        image_tensors = []
        context_ids = []
        context_attn_masks = []
        options_ids = []
        options_attn_masks = []
        # gt_relevances = []

        for data in data_list:
            image_ids.append(data["image_id"])
            image_tensor = data["image_tensor"]
            if isinstance(image_tensor, np.ndarray):
                image_tensor = torch.from_numpy(image_tensor)
            image_tensors.append(image_tensor)
            context_ids.append(data["text_ids"])
            context_attn_masks.append(data["attn_mask"])
            options_ids.append(data["options_ids"])
            options_attn_masks.append(data["options_attn_mask"])
            # gt_relevances.append(data['gt_relevance'])

        image_ids = torch.tensor(image_ids)
        image_tensors = torch.stack(image_tensors)
        num_image_per_seq = torch.ones(
            (image_tensors.shape[0],), dtype=torch.long, device=image_tensors.device
        )

        return dict(
            text_ids=context_ids,
            image_tensors=image_tensors,
            num_image_per_seq=num_image_per_seq,
            attention_mask=context_attn_masks,
            # gt_relevances=gt_relevances,
            options_ids=options_ids,
            options_attn_masks=options_attn_masks,
            image_ids=image_ids,
        )


class RICES:
    def __init__(
        self,
        dataset,
        batch_size,
        vision_encoder_path="./assets/openai/clip-vit-large-patch14",
        cached_features_path=None,
        image_size=224,
    ):
        self.dataset = dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.image_size = image_size

        # Load the model and processor
        self.model = CLIPModel.from_pretrained(vision_encoder_path)

        cached_features_path = os.path.join(
            cached_features_path, f"{dataset.__class__.__name__}.pth"
        )

        # Precompute features
        if cached_features_path is None or not os.path.exists(cached_features_path):
            self.model = self.model.to(self.device)
            self.features = self._precompute_features()
            self.model = self.model.to("cpu")
            if dist.get_rank() == 0:
                os.makedirs(os.path.dirname(cached_features_path), exist_ok=True)
                torch.save(self.features, cached_features_path)
            dist.barrier()
        else:
            self.features = torch.load(cached_features_path, map_location="cpu")

    def _precompute_features(self):
        features = []

        # Switch to evaluation mode
        self.model.eval()

        def custom_collate_fn(data_list):
            images = []
            for data in data_list:
                image = data[0]
                images.append(
                    torch.from_numpy(image) if isinstance(image, np.ndarray) else image
                )
            return torch.stack(images)

        # Set up loader
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
        )

        with torch.no_grad():
            for images in tqdm(
                loader,
                desc="Precomputing features for RICES",
            ):
                images = images.to(self.device)
                if images.shape[-1] != self.image_size:
                    images = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
                image_features = self.model.get_image_features(pixel_values=images)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                features.append(image_features.detach().cpu())

        features = torch.cat(features)
        return features

    def find(self, images, num_examples):
        """
        Get the top num_examples most similar examples to the images.
        """
        # Switch to evaluation mode
        self.model.eval()

        with torch.no_grad():
            if images.ndim == 3:
                images = images.unsqueeze(0)

            if images.shape[-1] != self.image_size:
                images = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            # Get the feature of the input image
            query_feature = self.model.get_image_features(pixel_values=images)
            query_feature = query_feature / query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()

            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]

        # Return with the most similar images last
        return [[self.dataset[i] for i in reversed(row)] for row in indices]
