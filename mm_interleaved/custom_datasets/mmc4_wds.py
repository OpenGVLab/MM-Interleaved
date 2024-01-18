from PIL import Image
import os.path as osp
from typing import Tuple
import functools
import json
import random
import math
import numpy as np

import braceexpand
import webdataset as wds

from transformers import LlamaTokenizer

from ..utils.misc import get_world_size
from .loader import BaseLoader
from .wds_utils import (
    jsonl_to_samples_nothrow,
    log_and_continue,
    SharedEpoch,
    ResampledShards2,
    detshuffle2,
    WdsDataset,
    interleaved_batched,
    extract_seq,
    concat_sample,
    init_tokenizer,
)
from .collator import interleaved_collation_fn

"""
    Iterable Version of MMC4, using webdataset for data processing
"""

Image.MAX_IMAGE_PIXELS = 1000000000
N_CHANNELS = 3
MIN_KB = 10
_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def dual_sort(a, b, reverse=False):
    # a is key by default
    c = list(zip(a,b))
    c.sort(key=lambda x:x[0], reverse=reverse)
    a, b = [t[0] for t in c], [t[1] for t in c]
    return a,b


def preprocess_mmc4_data(
    sample: Tuple[str],
    data_root="",
    transform=None,
    base_loader=None,
    tokenizer: LlamaTokenizer = None,
    num_total_token=2048,
    num_img_token=32,
    sim_threshold=0.1,
    max_num_images=6,
    min_num_images=1,
    img_first_prob=1.0,
):
    # add metadata for debug
    info, meta_info = json.loads(sample[0]), sample[-1]
    sentences = info["text_list"]
    sim_matrix = info["similarity_matrix"]

    # load PIL images and filter based on image-text similarity
    images, sentence_ixs = [], []
    for sample_image, sim_vec in zip(info["image_info"], sim_matrix):
        # bipartite matching as mmc4
        sim_ix = sample_image["matched_text_index"]
        sim_score = sample_image["matched_sim"]

        if sim_score < sim_threshold:
            continue

        image_name = sample_image["image_name"]
        image_path = osp.join(data_root, image_name)
        try:
            image = base_loader(image_path)
            image = image.convert("RGB")
        except:
            continue

        image = transform(image)
        images.append(image)
        sentence_ixs.append(sim_ix)

    if len(images) == 0:
        raise ValueError(f"Found no image in sample")

    keep_ixs = list(range(len(images)))
    random.shuffle(keep_ixs)
    keep_ixs = keep_ixs[:max_num_images]
    sentence_ixs = [sentence_ixs[ix] for ix in keep_ixs]
    images = [images[ix] for ix in keep_ixs]

    # make sure the image tensors have the same order as the image tokens in sequence
    sentence_ixs, images = dual_sort(sentence_ixs, images)

    if isinstance(images[0], tuple):
        # dual transform for encoder and decoder
        images_enc = [img[0] for img in images]
        images_dec = [img[1] for img in images]
        image_tensors = np.stack(images_enc, axis=0)
        image_tensors_dec = np.stack(images_dec, axis=0)
    else:
        image_tensors = np.stack(images, axis=0)
        image_tensors_dec = None
    num_images = image_tensors.shape[0]

    image_subseq = "<|beginofimage|>" + "<|image|>" * num_img_token
    for ix in sentence_ixs:
        # randomly place <|image|> before or after corresponding text
        img_first = np.random.random() < img_first_prob
        if img_first:
            sentences[ix] = image_subseq + sentences[ix]
        else:
            sentences[ix] = sentences[ix] + image_subseq

    text = " ".join(sentences)
    # whitespace cleanup
    text = (
        text
        .replace("<|image|> ", "<|image|>")
        .replace(" <|image|>", "<|image|>")
        .replace(" <|beginofimage|>", "<|beginofimage|>")
        .replace("<|beginofimage|> ", "<|beginofimage|>")
    )
    text = f"{text}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text,
        max_length=num_total_token,
        truncation=False,
        padding="do_not_pad",
        return_tensors="np",
        return_attention_mask=True,
    )
    text_ids = text_tensor["input_ids"][0]
    text_attn_mask = text_tensor["attention_mask"][0]


    if num_images < min_num_images:
        raise ValueError(f"Fewer than {min_num_images} images in sample")
    elif (
        num_images == 1 and random.random() <= 0.5
    ):  # 50% chance of keeping single image samples
        raise ValueError(f"Only one image in sample")

    image_tensors = image_tensors[:num_images]
    if image_tensors_dec is not None:
        image_tensors_dec = image_tensors_dec[:num_images]

    data = dict(
        image_tensors=image_tensors,
        text_ids=text_ids,
        text_attn_mask=text_attn_mask,
        image_tensors_dec=image_tensors_dec,
    )

    return data


def build_interleaved_dataset(
    shard_to_sample_fn,
    preprocess_fn,
    tokenizer,
    input_shards="docs_shard_{0..10}_v2.jsonl",
    per_device_batch_size=32,
    num_samples=None,
    resampled=False,
    floor=False,
    seed=42,
    epoch=0,
    num_workers=12,
    num_total_token=2048,
    num_img_token=32,
    max_num_images_per_seq=-1,
    loss_img_weight=None,
    loss_txt_weight=None,
    truncation_level="image",
    use_few_shot_sample=None,
    use_few_shot_prob=0.5,
):
    if not num_samples:
        raise RuntimeError(
            "Currently, number of dataset samples must be specified for training dataset. "
            "Please specify via `--train-num-samples` if no dataset length info present."
        )

    shards_list = list(braceexpand.braceexpand(input_shards))
    num_shards = len(shards_list)

    if not resampled:
        assert (
            num_shards >= num_workers * get_world_size()
        ), "number of shards must be >= total workers"

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)

    if resampled:
        pipeline = [
            ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )

    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            shard_to_sample_fn,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.map(preprocess_fn, handler=log_and_continue),
        ]
    )

    soi_token_id = tokenizer.additional_special_tokens_ids[
        tokenizer.additional_special_tokens.index("<|beginofimage|>")
    ]
    image_token_id = tokenizer.additional_special_tokens_ids[
        tokenizer.additional_special_tokens.index("<|image|>")
    ]
    sample_fn = functools.partial(
        extract_seq,
        num_total_token=num_total_token,
        num_img_token=num_img_token,
        max_num_images=max_num_images_per_seq,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        soi_token_id=soi_token_id,
        image_token_id=image_token_id,
        truncation_level=truncation_level,
        use_few_shot_sample=use_few_shot_sample,
        use_few_shot_prob=use_few_shot_prob,
    )
    pipeline.append(
        wds.pipelinefilter(concat_sample)(
            sample_fn=sample_fn, num_total_token=num_total_token, partial=False
        )
    )

    collate_fn = functools.partial(
        interleaved_collation_fn,
        pad_token_id=tokenizer.pad_token_id,
        return_nearest_bos_idxs=True,
        loss_img_weight=loss_img_weight,
        loss_txt_weight=loss_txt_weight,
    )

    pipeline.append(
        wds.pipelinefilter(interleaved_batched)(
            per_device_batch_size, collation_fn=collate_fn, partial=False
        )
    )

    dataset = WdsDataset(*pipeline)

    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = per_device_batch_size * get_world_size()
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, num_workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    # also return shared_epoch & num_samples & tokenizer
    dataset = dataset.with_length(num_samples)
    dataset.set_epoch(shared_epoch)
    dataset.set_tokenizer(tokenizer)

    print(
        f"num_samples: {num_samples} = global_batch_size: {global_batch_size} * num_batches: {num_batches}"
    )
    print(
        f"global_batch_size: {global_batch_size} = local_batch_size: {per_device_batch_size} * world_size: {get_world_size()}"
    )
    print(
        f"num_batches: {num_batches} = num_workers: {num_workers} * num_worker_batches: {num_worker_batches}"
    )

    return dataset


def build_mmc4_webdataset(
    annt_root="",
    data_root="",
    transform=None,
    tokenizer_path="",
    per_device_batch_size=32,
    input_shards="docs_shard_{0..10}_v2.jsonl",
    num_samples=None,
    resampled=False,
    floor=False,
    seed=42,
    epoch=0,
    num_workers=12,
    num_total_token=2048,
    num_img_token=64,
    max_num_images_per_seq=-1,
    sim_threshold=0.24,
    max_num_images=6,
    min_num_images=1,
    img_first_prob=0.5,
    loss_img_weight=None,
    loss_txt_weight=None,
    truncation_level="image",
):
    base_loader = BaseLoader()
    shard_to_sample_fn = functools.partial(
        jsonl_to_samples_nothrow,
        annt_root=annt_root,
        client=base_loader.client,
    )

    tokenizer = init_tokenizer(tokenizer_path)

    preprocess_fn = functools.partial(
        preprocess_mmc4_data,
        data_root=data_root,
        transform=transform,
        base_loader=base_loader,
        tokenizer=tokenizer,
        num_total_token=num_total_token,
        num_img_token=num_img_token,
        sim_threshold=sim_threshold,
        max_num_images=max_num_images,
        min_num_images=min_num_images,
        img_first_prob=img_first_prob,
    )

    dataset = build_interleaved_dataset(
        shard_to_sample_fn,
        preprocess_fn,
        tokenizer,
        per_device_batch_size=per_device_batch_size,
        input_shards=input_shards,
        num_samples=num_samples,
        resampled=resampled,
        floor=floor,
        seed=seed,
        epoch=epoch,
        num_workers=num_workers,
        num_total_token=num_total_token,
        num_img_token=num_img_token,
        max_num_images_per_seq=max_num_images_per_seq,
        loss_img_weight=loss_img_weight,
        loss_txt_weight=loss_txt_weight,
        truncation_level=truncation_level,
    )

    return dataset


if __name__ == "__main__":
    from .utils import create_transform

    transform = create_transform(
        aug_type="flip", resolution=256, random_crop=False, random_flip=True
    )

    dataset = build_mmc4_webdataset(
        data_root="./assets/mmc4/ai2-jackh-mmc4-gated-public-41423/images/",
        annt_root="./assets/mmc4/ai2-jackh-mmc4-gated-public-41423/data/",
        transform=transform,
        tokenizer_path="./assets/openlm-research/open_llama_3b_v2",
        per_device_batch_size=32,
        input_shards="docs_shard_{0..10}_v2.jsonl",
        num_samples=1000,
        resampled=False,
        floor=False,
        seed=42,
        num_workers=1,
        sim_threshold=0.1,
        max_num_images=100,
        min_num_images=1,
        num_img_token=32,
        num_total_token=2048,
        img_first_prob=0.5,
    )

    assert isinstance(dataset, wds.DataPipeline)
    print(dataset)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
    )
    print(dataloader)

    for i, data in enumerate(dataloader):
        print(f"iter: {i}")
        images_tensors, text_ids, text_attn_mask, num_images = (
            data["image_tensors"],
            data["text_ids"],
            data["attention_mask"],
            data["num_image_per_seq"],
        )
        texts = dataset.tokenizer.batch_decode(text_ids)
        break
