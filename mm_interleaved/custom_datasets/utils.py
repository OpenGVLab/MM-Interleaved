import torchvision.transforms as transforms
import numpy as np
import math
import random
from PIL import Image


from .mmc4_wds import build_mmc4_webdataset
from .laion_wds import build_laion_webdataset
from .mix_dataset import RandomMixWdsDataset
from .mscoco import CocoCaptionDataset
from .mscoco_karpathy import CocoCaptionKarpathyDataset
from .caption_datasets import NoCapsDataset, Flickr30KDataset
from .image2paragraph import Image2ParagraphDataset
from .visdial_dense import VisDialDenseDataset
from .lncoco import LNCOCODataset
from .vqa_datasets import (
    VQAV2Dataset,
    OKVQADataset,
    VizWizVQADataset,
    TextVQADataset,
)
from .grounding_datasets import (
    GroundingDataset,
    IterableKosmos2Dataset,
    RegionCaptionDataset,
    GroundedCaptionDataset,
    DatasetWrapper,
)
from .vist import VISTDataset
from .pororo import PororoDataset
from .flintstones import FlintStonesDataset
from .ade20k import ADE20kDataset
from .sft_datasets import WeightedConcatDataset
from .sft_datasets import LLaVADataset

from .collator import build_data_collator


def build_dataset(config):
    if isinstance(config, list):
        datasets = {}
        for _config in config:
            datasets[_config.name] = _build_dataset(_config)
        return datasets
    elif config.name == "random_mix":
        datasets = []
        for _config in config.datasets:
            datasets.append(_build_dataset(_config))
        dataset = RandomMixWdsDataset(
            datasets=datasets,
            probs=getattr(config, "probs", None),
            sampling_type=getattr(config, "sampling_type", "sum"),
            seed=getattr(config, "seed", 0),
            fix_sampling_ratio=getattr(config, "fix_sampling_ratio", False),
            dataset_names=getattr(config, "dataset_names", None),
        )
        dataset.collator = None
        return dataset

    return _build_dataset(config)


def _build_dataset(config):
    transform = create_transform(**config.transform)

    if config.name == "coco":
        dataset = CocoCaptionDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            phase=config.phase,
            year=config.year,
            total_length=getattr(config, "total_length", None),
            image_only=getattr(config, "image_only", True),
            collate_mode=getattr(config, "collate_mode", "generate_images"),
            rerank_by_clip=getattr(config, "rerank_by_clip", False),
        )
    elif config.name == "coco_karpathy":
        dataset = CocoCaptionKarpathyDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            phase=config.phase,
            year=config.year,
            total_length=getattr(config, "total_length", None),
            image_only=getattr(config, "image_only", True),
            collate_mode=getattr(config, "collate_mode", "generate_texts"),
            rerank_by_clip=getattr(config, "rerank_by_clip", False),
        )
    elif config.name == "image2paragraph":
        dataset = Image2ParagraphDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            phase=config.phase,
            total_length=getattr(config, "total_length", None),
            collate_mode=getattr(config, "collate_mode", "generate_texts"),
            add_eos=getattr(config, "add_eos", None),
        )
    elif config.name == "lncoco":
        dataset = LNCOCODataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            phase=config.phase,
            image_only=getattr(config, "image_only", False),
            total_length=getattr(config, "total_length", None),
            collate_mode=getattr(config, "collate_mode", "generate_images"),
            add_eos=getattr(config, "add_eos", None),
        )
    elif config.name == "visdial":
        dataset = VisDialDenseDataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            tokenizer_path=config.tokenizer_path,
            total_length=getattr(config, "total_length", None),
            num_img_token=getattr(config, "num_img_token", 64),
            collate_mode=getattr(config, "collate_mode", "generate_scores"),
            phase=config.phase,
        )
    elif config.name == "vist":
        dataset = VISTDataset(
            annt_root=config.annt_root,
            data_root=config.data_root,
            transform=transform,
            phase=config.phase,
            total_length=getattr(config, "total_length", None),
            collate_mode=getattr(config, "collate_mode", "generate_images"),
            add_eos=getattr(config, "add_eos", None),
            num_img_token=getattr(config, "num_img_token", 64),
            img_first_prob=getattr(config, "img_first_prob", 0.0),
            add_soi_token=getattr(config, "add_soi_token", True),
            round_range=config.round_range,
            context_type=config.context_type,
        )
    elif config.name == "pororo":
        dataset = PororoDataset(
            annt_root=config.annt_root,
            data_root=config.data_root,
            transform=transform,
            phase=config.phase,
            collate_mode=getattr(config, "collate_mode", "generate_images"),
            add_eos=getattr(config, "add_eos", None),
            num_img_token=getattr(config, "num_img_token", 64),
            img_first_prob=getattr(config, "img_first_prob", 0.0),
            add_soi_token=getattr(config, "add_soi_token", True),
            context_type=config.context_type,
            target_image_idxs=getattr(config, "target_image_idxs", None),
        )
    elif config.name == "flintstones":
        dataset = FlintStonesDataset(
            annt_root=config.annt_root,
            data_root=config.data_root,
            transform=transform,
            phase=config.phase,
            collate_mode=getattr(config, "collate_mode", "generate_images"),
            add_eos=getattr(config, "add_eos", None),
            num_img_token=getattr(config, "num_img_token", 64),
            img_first_prob=getattr(config, "img_first_prob", 0.0),
            add_soi_token=getattr(config, "add_soi_token", True),
            context_type=config.context_type,
            target_image_idxs=getattr(config, "target_image_idxs", None),
        )
    elif config.name == "mmc4_wds":
        # Iterable dataset
        dataset = build_mmc4_webdataset(
            annt_root=config.annt_root,
            data_root=config.data_root,
            transform=transform,
            tokenizer_path=config.tokenizer_path,
            per_device_batch_size=config.per_device_batch_size,
            input_shards=config.input_shards,
            num_samples=config.num_samples,
            floor=getattr(config, "floor", False),
            seed=getattr(config, "seed", 42),
            num_workers=getattr(config, "num_workers", 1),
            num_img_token=config.num_img_token,
            max_num_images_per_seq=getattr(config, "max_num_images_per_seq", -1),
            loss_img_weight=getattr(config, "loss_img_weight", None),
            loss_txt_weight=getattr(config, "loss_txt_weight", None),
        )
    elif config.name == "laion_wds":
        # Iterable dataset
        dataset = build_laion_webdataset(
            annt_root=config.annt_root,
            data_root=config.data_root,
            transform=transform,
            tokenizer_path=config.tokenizer_path,
            per_device_batch_size=config.per_device_batch_size,
            input_shards=config.input_shards,
            num_samples=config.num_samples,
            floor=getattr(config, "floor", False),
            seed=getattr(config, "seed", 42),
            num_workers=getattr(config, "num_workers", 1),
            num_img_token=config.num_img_token,
            max_num_images_per_seq=getattr(config, "max_num_images_per_seq", -1),
            loss_img_weight=getattr(config, "loss_img_weight", None),
            loss_txt_weight=getattr(config, "loss_txt_weight", None),
        )
    elif config.name == "nocaps":
        dataset = NoCapsDataset(
            data_root=config.data_root,
            annt_file=config.annt_file,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            image_only=getattr(config, "image_only", True),
            collate_mode=getattr(config, "collate_mode", "generate_texts"),
            add_eos=getattr(config, "add_eos", None),
        )
    elif config.name == "flickr30k":
        dataset = Flickr30KDataset(
            data_root=config.data_root,
            annt_file=config.annt_file,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            image_only=getattr(config, "image_only", True),
            collate_mode=getattr(config, "collate_mode", "generate_texts"),
            add_eos=getattr(config, "add_eos", None),
        )
    elif config.name == "vqav2":
        dataset = VQAV2Dataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            phase=getattr(config, "phase", "val"),
            collate_mode="generate_vqa",
            add_eos=getattr(config, "add_eos", None),
        )
    elif config.name == "okvqa":
        dataset = OKVQADataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            phase=getattr(config, "phase", "val"),
            collate_mode="generate_vqa",
            add_eos=getattr(config, "add_eos", None),
        )
    elif config.name == "vizwiz_vqa":
        dataset = VizWizVQADataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            phase=getattr(config, "phase", "val"),
            collate_mode="generate_vqa",
            add_eos=getattr(config, "add_eos", None),
            batch_size=getattr(config, "batch_size", 4),
        )
    elif config.name == "textvqa":
        dataset = TextVQADataset(
            data_root=config.data_root,
            annt_root=config.annt_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            phase=getattr(config, "phase", "val"),
            collate_mode="generate_vqa",
            add_eos=getattr(config, "add_eos", None),
        )
    elif config.name == "llava_instruct":
        dataset = LLaVADataset(
            annt_root=config.annt_root,
            data_root=config.data_root,
            transform=transform,
        )
    elif config.name == "weighted_concat_dataset":
        datasets, lengths = [], []
        for annt_item, data_item in zip(config.annt_root, config.data_root):
            dataset = LLaVADataset(
                annt_root=[annt_item],
                data_root=[data_item],
                transform=transform,
            )
            datasets.append(dataset)
            lengths.append(math.sqrt(len(dataset)))
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        for ds_name, weight in zip(config.annt_root, weights):
            print(f"{ds_name}: {weight}")
        dataset = WeightedConcatDataset(datasets, weights)
    elif config.name == "ade20k":
        dataset = ADE20kDataset(
            annt_root=config.annt_root,
            data_root=config.data_root,
            transform=transform,
            total_length=getattr(config, "total_length", None),
            phase=config.phase,
            collate_mode=getattr(config, "collate_mode", "generate_images"),
            add_eos=getattr(config, "add_eos", None),
            num_img_token=getattr(config, "num_img_token", 64),
            add_soi_token=getattr(config, "add_soi_token", True),
            text_first=getattr(config, "text_first", False),
            context_type=config.context_type,
        )
    elif config.name in (
        "vg",
        "refcocog_caption",
        "vg_test",
        "refcocog_caption_val",
        "refcocog_caption_train_val",
    ):
        dataset = RegionCaptionDataset(
            annt_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
            collate_mode="generate_texts",
            distributed=True,
            dataset_name=config.name,
        )
    elif config.name == "vgvqa":
        dataset = GroundingDataset(
            annt_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
            query_key="question",
            answer_key="answer",
            collate_mode="generate_texts",
            distributed=True,
            dataset_name=config.name,
        )
    elif config.name in (
        "refcoco_train_val",
        "refcoco",
        "refcoco_val",
        "refcoco_testA",
        "refcoco_testB",
        "refcoco+",
        "refcoco+_val",
        "refcoco+_testA",
        "refcoco+_testB",
        "refcocog",
        "refcocog_val",
        "refcocog_test",
    ):
        dataset = GroundingDataset(
            annt_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
            answer_key="sent",
            collate_mode="generate_grounding",
            distributed=not ("val" in config.name or "test" in config.name),
            dataset_name=config.name,
            random_flip=getattr(config, "random_flip", False),
            random_resize_crop_prob=getattr(config, "random_resize_crop_prob", 0.0),
        )
    elif config.name == "grit_grounding":
        dataset = IterableKosmos2Dataset(
            annt_root=config.annt_root,
            data_root=config.data_root,
            answer_key="sent",
            confidence_threshold=getattr(config, "confidence_threshold", 0),
            start_idx=getattr(config, "start_idx", 0),
            end_idx=getattr(config, "end_idx", 1),
            dataset_len=getattr(config, "dataset_len", 1),
            transform=transform,
            collate_mode="generate_grounding",
            distributed=True,
        )
    elif config.name in ("grit", "flickr30k_entities"):
        dataset = GroundedCaptionDataset(
            annt_file=config.annt_file,
            data_root=config.data_root,
            transform=transform,
            collate_mode="generate_texts",
            distributed=True,
            dataset_name=config.name,
        )
    else:
        raise NotImplementedError(config.name)

    if getattr(config, "train_dataset_config", None):
        collator = build_data_collator(
            config, train_dataset=build_dataset(config.train_dataset_config)
        )
    else:
        collator = build_data_collator(config)
    dataset.collator = collator
    dataset.dataset_name = config.name
    if not hasattr(dataset, "tokenizer"):
        setattr(dataset, "tokenizer", dataset.collator.tokenizer)

    if config.name in (
        "vg",
        "refcocog_caption",
        "vgvqa",
        "refcoco",
        "refcoco+",
        "refcocog",
        "grit",
        "flickr30k_entities",
        "grit_grounding",
    ):
        dataset = DatasetWrapper(
            dataset=dataset,
            concat_mode=getattr(config, "concat_mode", False),
            per_device_batch_size=config.per_device_batch_size,
        )

    return dataset


def create_transform(
    aug_type="numpy",
    resolution=224,
    resize=True,
    random_crop=False,
    center_crop=True,
    random_flip=False,
    neg_normalize=False,
    scale=None,
    resolution2=512,
):
    if aug_type == "numpy":
        assert resize
        transform = transform_numpy(
            resolution=resolution,
            random_crop=random_crop,
            center_crop=center_crop,
            random_flip=random_flip,
            neg_normalize=neg_normalize,
        )
    elif aug_type == "flip":
        assert not random_crop
        transform = []
        if resize:
            resize_size = max(256, resolution)
            transform.append(
                transforms.Resize(
                    resize_size, interpolation=transforms.InterpolationMode.BICUBIC
                )
            )
        transform.append(transforms.CenterCrop(resolution))
        if random_flip:
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        transform = transforms.Compose(transform)
    elif aug_type.startswith("dual_"):
        aug_type = aug_type.replace("dual_", "")
        assert resolution2 > 0, f"{aug_type=}; {resolution2=}"
        transform = dual_transform(
            resolution1=resolution,
            resolution2=resolution2,
            aug_type=aug_type,
            resize=resize,
            random_crop=random_crop,
            random_flip=random_flip,
            neg_normalize=neg_normalize,
            scale=scale,
        )
    elif aug_type == "resize":
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (resolution, resolution),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
            ]
        )
    elif aug_type == "numpy_grounding":
        assert resize
        transform = transform_numpy_grounding(
            resolution=resolution,
            neg_normalize=neg_normalize,
        )
    else:
        raise NotImplementedError
    return transform


class dual_transform:
    def __init__(
        self,
        resolution1,
        resolution2,
        aug_type="numpy",
        resize=False,
        random_crop=False,
        random_flip=True,
        neg_normalize=True,
        scale=0.2,
    ):
        self.transform1 = create_transform(
            aug_type=aug_type,
            resolution=resolution1,
            resize=resize,
            random_crop=random_crop,
            random_flip=random_flip,
            neg_normalize=neg_normalize,
            scale=scale,
            resolution2=-1,
        )

        self.transform2 = create_transform(
            aug_type=aug_type,
            resolution=resolution2,
            resize=resize,
            random_crop=random_crop,
            random_flip=random_flip,
            neg_normalize=neg_normalize,
            scale=scale,
            resolution2=-1,
        )

    def __call__(self, pil_image):
        arr1 = self.transform1(pil_image)
        arr2 = self.transform2(pil_image)

        return arr1, arr2

    def __repr__(self):
        return f"Dual Transform: {self.transform1}\n{self.transform2}"


class transform_numpy:
    def __init__(
        self,
        resolution,
        random_crop=False,
        center_crop=True,
        random_flip=True,
        neg_normalize=True,
    ) -> None:
        self.resolution = resolution
        self.random_crop = random_crop
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.neg_normalize = neg_normalize

    def __call__(self, pil_image):
        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            if self.center_crop:
                arr = center_crop_arr(pil_image, self.resolution)
            else:
                arr = np.array(
                    pil_image.resize(
                        (self.resolution, self.resolution), resample=Image.BICUBIC
                    )
                )

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32).transpose([2, 0, 1])
        if self.neg_normalize:
            # normalize to [-1,1]
            arr = arr / 127.5 - 1
        else:
            # normalize to [0,1]
            arr = arr / 255.0
        return arr

    def __repr__(self):
        return (
            f"transform_numpy: {self.resolution=}, {self.random_crop=}, "
            f"{self.random_flip=}, {self.neg_normalize=}"
        )


class transform_numpy_grounding:
    def __init__(self, resolution, neg_normalize=True) -> None:
        self.resolution = resolution
        self.neg_normalize = neg_normalize

    def __call__(self, pil_image):
        arr = resize_arr(pil_image, self.resolution)
        arr = arr.astype(np.float32).transpose([2, 0, 1])
        if self.neg_normalize:
            # normalize to [-1,1]
            arr = arr / 127.5 - 1
        else:
            # normalize to [0,1]
            arr = arr / 255.0
        return arr

    def __repr__(self):
        return f"transform_numpy_grounding: {self.resolution=}, {self.neg_normalize=}"


def resize_arr(pil_image, image_size):
    pil_image = pil_image.resize((image_size, image_size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    return arr


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def _smart_join(str_or_list, delim):
    if isinstance(str_or_list, str):
        return str_or_list
    else:
        return delim.join(str_or_list)
