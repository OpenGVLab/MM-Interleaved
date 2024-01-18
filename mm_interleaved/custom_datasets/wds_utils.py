"""
Util functions for initializing webdataset objects
"""

import ast
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from multiprocessing import Value

import braceexpand
import numpy as np
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

LOG_WDS = os.environ.get("LOG_WDS", False)


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards[0])
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum(
            [
                int(sizes[os.path.basename(shard)])
                if os.path.basename(shard) in sizes
                else 0
                for shard in shards_list
            ]
        )
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, "r").read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def log_and_continue(exn, force=False):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    if LOG_WDS or force:
        logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)

        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


from webdataset.utils import pytorch_worker_info
import io
import os.path as osp
import zipfile
from transformers import AutoTokenizer


def jsonl_to_samples_nothrow(src, annt_root="", client=None, handler=log_and_continue):
    rank, world_size, worker, num_workers = pytorch_worker_info()
    for sample in src:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        jsonl_fname = sample["url"]
        try:
            data_path = osp.join(annt_root, f"{jsonl_fname}.zip")
            print(
                f"[Rank {rank:02d} Worker {worker:02d}] start load from {data_path}",
                force=True,
            )

            with io.BytesIO(client.get(data_path)) as rf:
                with zipfile.ZipFile(rf) as zrf:
                    with io.BytesIO(zrf.read(jsonl_fname)) as jrf:
                        lines = jrf.readlines()

            for i, line in enumerate(lines):
                yield (line, f"{jsonl_fname}-{i}")

            print(
                f"[Rank {rank:02d} Worker {worker:02d}] finish load from {data_path}",
                force=True,
            )
        except Exception as exn:
            import traceback

            traceback.print_stack()
            exn.args = exn.args + (jsonl_fname,)
            if handler(exn, force=True):
                continue
            else:
                break


def init_tokenizer(tokenizer_path, add_grounding_special_tokens=False):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False
    )  # vocab_size: 32000
    all_special_tokens = tokenizer.all_special_tokens
    if "decapoda-research" in tokenizer_path:
        # begin of sequence, end of sequence
        all_special_tokens += ["<s>", "</s>"]
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 31999
    elif "openlm-research" in tokenizer_path:
        tokenizer.pad_token_id = 31999
    elif "vicuna" in tokenizer_path:
        tokenizer.pad_token_id = 31999
    elif "llama" in tokenizer_path:
        tokenizer.pad_token_id = 31999
    # end of chunk, begin of image, image
    # all_special_tokens += ['<|endofchunk|>', '<|beginofimage|>', '<|image|>']
    all_special_tokens += ["<|beginofimage|>", "<|image|>"]
    if add_grounding_special_tokens:
        all_special_tokens += [
            "<ref>",
            "</ref>",
            "<box>",
            "</box>",
        ]  # This is <ref>a dog</ref><box>(x1,y1)(x2,y2)</box>
    special_tokens_dict = {"additional_special_tokens": all_special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer

class WdsDataset(wds.DataPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch: SharedEpoch = None
        self.tokenizer = None

    def set_epoch(self, epoch):
        if isinstance(epoch, int):
            self.epoch.set_value(epoch)
        elif isinstance(epoch, SharedEpoch):
            self.epoch = epoch
        else:
            raise ValueError(f"unsupported epoch type: {type(epoch)}")

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __repr__(self) -> str:
        repr_str = f"WdsDataset: num_samples={len(self)}; epoch={self.epoch}; "
        repr_str += f"pipeline={self.pipeline}"
        return repr_str


def extract_data_from_buffer(buffers, num_total_token, num_images):
    text_ids, buffers["text_ids"] = (
        buffers["text_ids"][:num_total_token],
        buffers["text_ids"][num_total_token:],
    )

    text_attn_mask, buffers["text_attn_mask"] = (
        buffers["text_attn_mask"][:num_total_token],
        buffers["text_attn_mask"][num_total_token:],
    )

    image_tensors, buffers["image_tensors"] = (
        buffers["image_tensors"][:num_images],
        buffers["image_tensors"][num_images:],
    )

    if buffers.get("image_tensors_dec", None) is not None:
        image_tensors_dec, buffers["image_tensors_dec"] = (
            buffers["image_tensors_dec"][:num_images],
            buffers["image_tensors_dec"][num_images:],
        )
    else:
        image_tensors_dec, buffers["image_tensors_dec"] = None, None

    data = dict(
        image_tensors=image_tensors,
        text_ids=text_ids,
        text_attn_mask=text_attn_mask,
        image_tensors_dec=image_tensors_dec,
    )

    return data, buffers


def calc_nearest_bos_token_idxs(
    text_ids,
    bos_token_id=1,
    soi_token_id=32001,
):
    soi_token_idxs = (text_ids == soi_token_id).nonzero()[0]
    try:
        bos_token_idxs = (text_ids == bos_token_id).nonzero()[0]
        bos_token_idxs = np.insert(bos_token_idxs, 0, 0, axis=0)
        nearest_bos_idxs = []
        for soi_token_idx in soi_token_idxs:
            if soi_token_idx == 0:
                nearest_bos_idx = 0
            else:
                nearest_bos_idx = bos_token_idxs[
                    (bos_token_idxs < soi_token_idx).nonzero()[0].max()
                ]
            nearest_bos_idxs.append(nearest_bos_idx)
        nearest_bos_idxs = np.array(nearest_bos_idxs)
    except Exception as exp:
        print(f"{text_ids=}", force=True)
        print(f"{soi_token_idxs=}, {bos_token_idxs=}", force=True)
        raise exp
    return nearest_bos_idxs


def check_image_truncate(
    data,
    buffers,
    num_img_token=32,
    bos_token_id=1,
    soi_token_id=32001,
    image_token_id=32002,
    truncation_level="image",
):
    soi_token_idxs = (data["text_ids"] == soi_token_id).nonzero()[0]
    if len(soi_token_idxs) > 0:
        last_soi_token_idx = soi_token_idxs[-1]
        if last_soi_token_idx >= len(data["text_ids"]) - num_img_token:
            data["meta"]["is_truncated"] = 1
            # truncation detected
            if truncation_level == "sample":
                bos_token_idxs = (data["text_ids"] == bos_token_id).nonzero()[0]
                last_bos_token_idx = bos_token_idxs[-1]
                data["text_ids"], text_ids_left = (
                    data["text_ids"][:last_bos_token_idx],
                    data["text_ids"][last_bos_token_idx:],
                )
                buffers["text_ids"] = np.concatenate(
                    (text_ids_left, buffers["text_ids"]), axis=0
                )
                data["text_attn_mask"], text_attn_mask_left = (
                    data["text_attn_mask"][:last_bos_token_idx],
                    data["text_attn_mask"][last_bos_token_idx:],
                )
                buffers["text_attn_mask"] = np.concatenate(
                    (text_attn_mask_left, buffers["text_attn_mask"]), axis=0
                )
                # soi_token_idxs = (data["text_ids"] == soi_token_id).nonzero()[0]
                num_images = (
                    np.count_nonzero(data["text_ids"] == image_token_id)
                    // num_img_token
                )
                data["image_tensors"], image_tensors_left = (
                    data["image_tensors"][:num_images],
                    data["image_tensors"][num_images:],
                )
                buffers["image_tensors"] = np.concatenate(
                    (image_tensors_left, buffers["image_tensors"]), axis=0
                )
                if data["image_tensors_dec"] is not None:
                    data["image_tensors_dec"], image_tensors_dec_left = (
                        data["image_tensors_dec"][:num_images],
                        data["image_tensors_dec"][num_images:],
                    )
                    buffers["image_tensors_dec"] = np.concatenate(
                        (image_tensors_dec_left, buffers["image_tensors_dec"]), axis=0
                    )
            else:
                data["text_ids"], text_ids_left = (
                    data["text_ids"][:last_soi_token_idx],
                    data["text_ids"][last_soi_token_idx:],
                )
                buffers["text_ids"] = np.concatenate(
                    (text_ids_left, buffers["text_ids"]), axis=0
                )
                data["text_attn_mask"], text_attn_mask_left = (
                    data["text_attn_mask"][:last_soi_token_idx],
                    data["text_attn_mask"][last_soi_token_idx:],
                )
                buffers["text_attn_mask"] = np.concatenate(
                    (text_attn_mask_left, buffers["text_attn_mask"]), axis=0
                )
                # soi_token_idxs = soi_token_idxs[:-1]

    return data, buffers


def update_meta_stats(
    data,
    bos_token_id=1,
    soi_token_id=32001,
):
    data["meta"]["is_first_token_image"] = int(
        (data["text_ids"][0] == soi_token_id)
        or (data["text_ids"][0] == bos_token_id and data["text_ids"][1] == soi_token_id)
    )
    soi_token_idxs = (data["text_ids"] == soi_token_id).nonzero()[0]
    data["meta"]["uncond_image_cnt"] = int(
        (soi_token_idxs - data["nearest_bos_idxs"] <= 1).sum()
    )
    data["meta"]["image_cnt"] = int(data["image_tensors"].shape[0])


def extract_seq(
    buffers,
    num_total_token=2048,
    num_img_token=32,
    max_num_images=-1,
    bos_token_id=1,
    eos_token_id=2,
    soi_token_id=32001,
    image_token_id=32002,
    truncation_level="image",
    use_few_shot_sample=None,
    use_few_shot_prob=0.5,
):
    assert truncation_level in ["image", "sample"]

    num_images = (
        np.count_nonzero(buffers["text_ids"][:num_total_token] == image_token_id)
        // num_img_token
    )
    if max_num_images > 0 and num_images > max_num_images:
        soi_token_idxs = (buffers["text_ids"] == soi_token_id).nonzero()[0]
        if truncation_level == "sample":
            # find the nearer token between next <|soi|> and next <|bos|>
            next_soi_token_idx = soi_token_idxs[max_num_images]
            last_bos_token_idx = (
                buffers["text_ids"][:next_soi_token_idx] == bos_token_id
            ).nonzero()[0][-1]
            if last_bos_token_idx > soi_token_idxs[max_num_images - 1]:
                num_total_token = last_bos_token_idx
            else:
                num_total_token = next_soi_token_idx
        else:
            num_total_token = soi_token_idxs[max_num_images - 1] + num_img_token + 1
        num_images = max_num_images

    data, buffers = extract_data_from_buffer(
        buffers, num_total_token=num_total_token, num_images=num_images
    )
    data["meta"] = dict(
        is_truncated=0,
    )

    # now we consider the image truncation case
    data, buffers = check_image_truncate(
        data,
        buffers,
        num_img_token=num_img_token,
        bos_token_id=bos_token_id,
        soi_token_id=soi_token_id,
        image_token_id=image_token_id,
        truncation_level=truncation_level,
    )

    if use_few_shot_sample is not None and np.random.rand() < use_few_shot_prob:
        few_shot_num = np.random.choice(use_few_shot_sample)
        bos_token_idxs = (data["text_ids"] == bos_token_id).nonzero()[0]
        eos_token_idxs = (bos_token_idxs - 1).clip(min=0)
        bos_token_idxs_reserve = bos_token_idxs[::few_shot_num]
        eos_token_idxs_reserve = eos_token_idxs[::few_shot_num]

        text_mask = np.ones(len(data["text_ids"]), dtype=np.bool_)
        text_mask[bos_token_idxs] = False
        text_mask[eos_token_idxs] = False
        text_mask[bos_token_idxs_reserve] = True
        text_mask[eos_token_idxs_reserve] = True
        text_mask = np.logical_or(text_mask, ~np.isin(text_mask, [bos_token_id, eos_token_id]))
        
        data["text_ids"] = data["text_ids"][text_mask]
        data["text_attn_mask"] = data["text_attn_mask"][text_mask]
        
    # find the nearest <bos> token of each image
    if num_images > 0:
        nearest_bos_idxs = calc_nearest_bos_token_idxs(
            text_ids=data["text_ids"],
            bos_token_id=bos_token_id,
            soi_token_id=soi_token_id,
        )
        data["nearest_bos_idxs"] = nearest_bos_idxs

    if num_images > 0:
        update_meta_stats(data, bos_token_id=bos_token_id, soi_token_id=soi_token_id)

    if num_images <= 0:
        data = None

    return data, buffers


def concat_sample(
    data,
    sample_fn=extract_seq,
    num_total_token=2048,
    partial=False,
):
    """
    Maintain a data buffer for the training samples,
    each time when calling it will yield a sequence using `sample_fn` whose length is num_total_token
    """
    buffers = dict(
        text_ids=None,
        text_attn_mask=None,
        image_tensors=None,
        image_tensors_dec=None,
    )

    for sample in data:
        while (
            buffers["text_ids"] is not None
            and len(buffers["text_ids"]) >= num_total_token
        ):
            sample_n, buffers_n = sample_fn(buffers)
            buffers = buffers_n
            if sample_n is not None:
                yield sample_n

        if buffers["text_ids"] is None:
            for k, v in sample.items():
                if v is not None:
                    buffers[k] = v.copy()
        else:
            for k, v in sample.items():
                if v is not None:
                    buffers[k] = np.concatenate((buffers[k], v), axis=0)

    if buffers["text_ids"] is None or len(buffers["text_ids"]) == 0:
        return
    elif len(buffers["text_ids"]) == num_total_token or partial:
        sample_n, _ = sample_fn(buffers)
        if sample_n is not None:
            yield sample_n


def interleaved_batched(
    data,
    batchsize=20,
    collation_fn=None,
    partial=True,
):
    """Create batches of the given size.

    :param data: iterator
    :param batchsize: target batch size
    :param tensors: automatically batch lists of ndarrays into ndarrays
    :param partial: return partial batches
    :returns: iterator

    """
    batch = []
    for sample in data:
        if len(batch) >= batchsize:
            if collation_fn is not None:
                batch = collation_fn(batch)
            yield batch
            batch = []

        batch.append(sample)
    if len(batch) == 0:
        return
    elif len(batch) == batchsize or partial:
        if collation_fn is not None:
            batch = collation_fn(batch)
        yield batch


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.
        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch

        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))
