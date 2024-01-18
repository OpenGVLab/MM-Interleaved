import io
from PIL import Image
import os.path as osp
from typing import Tuple
import json
import numpy as np
import functools

import webdataset as wds
from webdataset.utils import pytorch_worker_info

from transformers import LlamaTokenizer


from .loader import BaseLoader
from .wds_utils import (
    init_tokenizer,
    log_and_continue,
)
from .mmc4_wds import build_interleaved_dataset

"""
    Iterable Version of LAION, using webdataset for data processing
"""

Image.MAX_IMAGE_PIXELS = 1000000000
N_CHANNELS = 3
MIN_KB = 10


def load_laion_database_nothrow(
    src,
    annt_root="",
    handler=log_and_continue,
    client=None,
):
    rank, world_size, worker, num_workers = pytorch_worker_info()
    for sample in src:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        annt_fname = sample["url"]
        data_path = osp.join(annt_root, annt_fname)

        try:
            print(
                f"[Rank {rank:02d} Worker {worker:02d}] start load from {data_path}",
                force=True,
            )

            with io.BytesIO(client.get(data_path)) as rf:
                lines = rf.readlines()

            for i, line in enumerate(lines):
                yield (line, f"{annt_fname}-{i}")

            print(
                f"[Rank {rank:02d} Worker {worker:02d}] finish load from {data_path}",
                force=True,
            )

        except Exception as exn:
            import traceback

            traceback.print_stack()
            exn.args = exn.args + (data_path,)
            if handler(exn, force=True):
                continue
            else:
                break


def _smart_join(str_or_list, delim):
    if isinstance(str_or_list, str):
        return str_or_list
    else:
        return delim.join(str_or_list)


def preprocess_laion_data(
    sample: Tuple[str],
    data_root="",
    transform=None,
    base_loader=None,
    tokenizer: LlamaTokenizer = None,
    num_total_token=2048,
    num_img_token=32,
    img_first_prob=1.0,
):
    info, meta_info = json.loads(sample[0]), sample[-1]
 
    image_name = info["image"]
    image_path = osp.join(data_root, image_name)
    try:
        image = base_loader(image_path)
        image = image.convert("RGB")
    except:
        raise ValueError(f"Failed to load Image {image_path}")

    image_tensors = transform(image)

    if isinstance(image_tensors, tuple):
        image_tensors, image_tensors_dec = np.expand_dims(
            image_tensors[0], axis=0
        ), np.expand_dims(image_tensors[1], axis=0)
    else:
        image_tensors, image_tensors_dec = np.expand_dims(image_tensors, axis=0), None

    img_first = np.random.random() < img_first_prob
    caption = _smart_join(info["caption"], " ").lower()
    image_subseq = "<|beginofimage|>" + "<|image|>" * num_img_token

    if img_first:
        text = image_subseq + caption
    else:
        text = caption + image_subseq

    text = f"{text}{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text,
        padding="do_not_pad",
        return_tensors="np",
        return_attention_mask=True,
    )

    text_ids = text_tensor["input_ids"][0]
    text_attn_mask = text_tensor["attention_mask"][0]

    if len(text_ids) > num_total_token:
        if img_first:
            text_ids = text_ids[:num_total_token]
            text_attn_mask = text_attn_mask[:num_total_token]
        else:
            text_ids = np.concatenate(
                (
                    text_ids[: num_total_token - (num_img_token + 2)],
                    text_ids[-(num_img_token + 2) :],
                ),
                axis=0,
            )
            text_attn_mask = np.concatenate(
                (
                    text_attn_mask[: num_total_token - (num_img_token + 2)],
                    text_attn_mask[-(num_img_token + 2) :],
                ),
                axis=0,
            )

    data = dict(
        image_tensors=image_tensors,
        text_ids=text_ids,
        text_attn_mask=text_attn_mask,
        image_tensors_dec=image_tensors_dec,
    )

    return data


def build_laion_webdataset(
    annt_root="",
    data_root="",
    transform=None,
    tokenizer_path="",
    per_device_batch_size=32,
    input_shards="{0000000..0000010}.txt",
    num_samples=None,
    resampled=False,
    floor=False,
    seed=42,
    epoch=0,
    num_workers=12,
    num_total_token=2048,
    num_img_token=64,
    max_num_images_per_seq=-1,
    img_first_prob=0.5,
    loss_img_weight=None,
    loss_txt_weight=None,
    truncation_level="sample",
    use_few_shot_sample=[2,3,4,5,6,7,8],
    use_few_shot_prob=0.25,
):
    base_loader = BaseLoader()
    shard_to_sample_fn = functools.partial(
        load_laion_database_nothrow,
        annt_root=annt_root,
        client=base_loader.client,
    )

    tokenizer = init_tokenizer(tokenizer_path)

    preprocess_fn = functools.partial(
        preprocess_laion_data,
        data_root=data_root,
        transform=transform,
        base_loader=base_loader,
        tokenizer=tokenizer,
        num_total_token=num_total_token,
        num_img_token=num_img_token,
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
        use_few_shot_sample=use_few_shot_sample,
        use_few_shot_prob=use_few_shot_prob,
    )

    return dataset


if __name__ == "__main__":
    from .utils import create_transform

    transform = create_transform(
        aug_type="numpy",
        resolution=256,
        resize=True,
        random_crop=False,
        random_flip=True,
    )

    dataset = build_laion_webdataset(
        annt_root="./assets/laion5b/LaionEn",
        data_root="",
        transform=transform,
        tokenizer_path="./assets/openlm-research/open_llama_3b_v2",
        per_device_batch_size=32,
        input_shards="{0000000..0010336}.txt",
        num_samples=2_600_000,
        resampled=False,
        floor=False,
        seed=42,
        num_workers=1,
        num_img_token=32,
        max_num_images_per_seq=-1,
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
        images_tensors, text_ids, text_attn_mask, num_images = (
            data["image_tensors"],
            data["text_ids"],
            data["attention_mask"],
            data["num_image_per_seq"],
        )
        texts = dataset.tokenizer.batch_decode(text_ids)

        print(images_tensors.shape)
        print(text_ids)
        print(num_images)
        print(data["meta"])

        break
