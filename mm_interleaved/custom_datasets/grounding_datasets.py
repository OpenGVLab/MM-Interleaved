import re
import os
import json
import os
import json
import random
from PIL import Image
from typing import Iterator, Optional, List

import torch
import torchvision.transforms as T
import torch.distributed as dist
import torchvision.transforms.functional as F
from torch.utils.data import IterableDataset
from timm.data.transforms import RandomResizedCropAndInterpolation

from .loader import BaseDataset, IterableBaseDataset
from .collator import GroundingCollator


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def load_json(url, client):
    with open(url, 'r') as file:
        data = json.load(file)
    return data


class GroundingBaseDataset(BaseDataset):
    def __init__(
        self,
        transform: Optional[T.Compose] = None,
        box_scale: int = 999,
        collator: Optional[GroundingCollator] = None,
        dataset_name: str = None,
        collate_mode: str = 'generate_grounding',
        return_image: bool = True,
        random_flip: bool = False,
        random_resize_crop_prob: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ann = []
        self.box_scale = box_scale
        self.transform = transform
        self.collator = collator
        self.collate_mode = collate_mode
        self.return_image = return_image
        self.dataset_name = dataset_name if dataset_name is not None else self.__class__.__name__

        self.random_flip = random_flip
        self.random_resize_crop_prob = random_resize_crop_prob
        self.grounded_caption_err = 0

        if self.random_resize_crop_prob > 0:
            self.random_resize_crop = RandomResizedCrop(self.transform.resolution, interpolation='bicubic')

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx): 
        ann = self.ann[idx]

        data = {}
        data['id'] = ann['id']
        
        image = ann['image']
        data['image'] = self.loader(image).convert('RGB') if self.return_image else image

        if 'label' in ann:
            data['label'] = ann['label']
            
            if isinstance(self, GroundedCaptionDataset):
                try:
                    data['label'] = GroundedCaptionDataset.rescale_boxes(data['label'], data['image'].height, data['image'].width, self.box_scale)
                except:
                    self.grounded_caption_err += 1
                    print(f'[{self.__class__.__name__}] parse err, randomly return another sample (err_cnt: {self.grounded_caption_err})')
                    return self.__getitem__(random.randint(0, len(self)))

        if self.transform is not None and self.return_image:
            data['images_tensor'] = self.transform(data['image'])

        if 'query' in ann:
            data['query'] = ann['query']

        if 'bbox' in ann:
            x1, y1, x2, y2 = ann['bbox']
            assert x1 <= x2 and y1 <= y2, ann

            data['bbox'] = (
                x1 / data['image'].width * self.box_scale,
                y1 / data['image'].height * self.box_scale,
                x2 / data['image'].width * self.box_scale,
                y2 / data['image'].height * self.box_scale,
            )

        return self.data_augment(data)

    def distribute_ann(self):
        rank = get_rank()
        world_size = get_world_size()

        per_rank = len(self.ann) // world_size
        start_idx = rank * per_rank
        end_idx = start_idx + per_rank

        self.ann = self.ann[start_idx:end_idx]
        self.synchronize_len()

        log_info = ''
        log_info += f'[{self.dataset_name}] '
        log_info += f'Rank: {rank:02d}/{world_size:02d} keep ann from {start_idx}~{end_idx}, synchronized_len: {len(self.ann)}'
        print(log_info, force=True)

    def synchronize_len(self):
        if is_dist_avail_and_initialized():
            all_rank_len = [None] * get_world_size()
            dist.all_gather_object(all_rank_len, len(self))
            min_len = min(all_rank_len)
        else:
            min_len = len(self)

        self.ann = self.ann[:min_len]

    def shuffle(self):
        random.shuffle(self.ann)

    @staticmethod
    def allow_random_crop(caption):
        keywords = ['top', 'bottom', 'left', 'right', 'center', 'middle', 'above', 'below', 'first', 'second', 'third']
        for keyword in keywords:
            if keyword in caption:
                return False
        return True

    def data_augment(self, data):
        if self.random_flip and random.random() < 0.5:
            data['image'] = data['image'].transpose(Image.FLIP_LEFT_RIGHT)
            data['images_tensor'] = self.transform(data['image'])
            
            caption = data['label']
            caption = caption.replace('left', '<LEFT>')
            caption = caption.replace('right', '<RIGHT>')
            # print(f'[caption befor flip] {data["label"]}')
            data['label'] = caption.replace('<LEFT>', 'right').replace('<RIGHT>', 'left')
            # print(f'[caption after flip] {data["label"]}')

            x1, y1, x2, y2 = data['bbox']
            x1 = x1 / self.box_scale
            y1 = y1 / self.box_scale
            x2 = x2 / self.box_scale
            y2 = y2 / self.box_scale

            flip_x1 = 1 - x1
            flip_x2 = 1 - x2
            x1 = flip_x2
            x2 = flip_x1

            data['bbox'] = (
                x1 * self.box_scale,
                y1 * self.box_scale,
                x2 * self.box_scale,
                y2 * self.box_scale,
            )

        if self.allow_random_crop(data['label']) and random.random() < self.random_resize_crop_prob:
            image = data['image']
            x1, y1, x2, y2 = data['bbox']
            bbox = (
                x1 / self.box_scale * image.width,
                y1 / self.box_scale * image.height,
                x2 / self.box_scale * image.width,
                y2 / self.box_scale * image.height,
            )
            
            image, bbox = self.random_resize_crop(image, bbox)
            data['image'] = image
            data['images_tensor'] = self.transform(data['image'])
            
            x1, y1, x2, y2 = bbox
            bbox = (
                x1 / self.transform.resolution * self.box_scale,
                y1 / self.transform.resolution * self.box_scale,
                x2 / self.transform.resolution * self.box_scale,
                y2 / self.transform.resolution * self.box_scale,
            )
            data['bbox'] = bbox
            # print(f'[caption after random_resize_crop] {data["label"]}')

        x1, y1, x2, y2 = data['bbox']
        data['bbox'] = (int(x1), int(y1), int(x2), int(y2))

        return data


# jsonl format
class GroundingDataset(GroundingBaseDataset):
    def __init__(
        self,
        data_root: str,
        annt_file: str,
        answer_key: str,
        query_key: Optional[str] = None,
        distributed: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.annt_file = annt_file
        self.query_key = query_key
        self.answer_key = answer_key

        with open(self.annt_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            ann = json.loads(line)
            data = {
                'id': -1,
                'image': os.path.join(self.data_root, ann['image']),
                'label': ann[self.answer_key],
            }
            if self.query_key is not None:
                data['query'] = ann[self.query_key]
            if 'bbox' in ann:
                data['bbox'] = ann['bbox']

            self.ann.append(data)

        if distributed:
            self.distribute_ann()

        self.shuffle()


# coco format
class RegionCaptionDataset(GroundingBaseDataset):
    def __init__(
        self,
        data_root: str,
        annt_file: str,
        distributed: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.annt_file = annt_file

        annotations = load_json(annt_file, self.loader.client)['annotations']
        for ann in annotations:
            item = {
                'id': ann['image_id'],
                'image': os.path.join(data_root, ann['image']),
                'label': ann['caption'],
            }

            if 'query' in ann:
                item['query'] = ann['query']

            if 'bbox' in ann:
                # x1y1x2y2
                item['bbox'] = ann['bbox']

            self.ann.append(item)

        if distributed:
            self.distribute_ann()

        self.shuffle()

class GroundedCaptionDataset(GroundingBaseDataset):
    def __init__(
        self,
        data_root: str,
        annt_file: str,
        distributed: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.annt_file = annt_file

        with open(annt_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            ann = json.loads(line)
            self.ann.append({
                'id': -1,
                'image': os.path.join(data_root, ann['image']),
                'label': ann['sent'],
            })

        if distributed:
            self.distribute_ann()

        self.shuffle()

    @staticmethod
    def parse_box_str(box_str: str):
        # box_str: (x1,y1)(x2,y2)
        x1y1, x2y2 = re.findall(r'\((.*?)\)', box_str)
        x1, y1 = x1y1.split(',')
        x2, y2 = x2y2.split(',')

        return float(x1), float(y1), float(x2), float(y2)

    @staticmethod
    def extract_objects(
        grounded_caption: str,
        grounded_pattern: str = r'<.*?>.*?<.*?>',
        ref_tag: str = '<ref>',
        box_tag: str = '<box>',
    ):
        objects = {}
        res = re.findall(grounded_pattern, grounded_caption)

        last_item = None
        for item in res:
            clean_item = re.sub(r'<.*?>', '', item)
            if item.startswith(ref_tag):
                last_item = clean_item
                objects[last_item] = []
            else:
                assert item.startswith(box_tag), f'{item}\n{grounded_caption}'
                assert last_item is not None
                objects[last_item].append(clean_item)

        return objects

    @staticmethod
    def rescale_boxes(grounded_caption: str, height: int, width: int, scale: int):
        objects = GroundedCaptionDataset.extract_objects(grounded_caption)
        all_boxes = []
        for v in objects.values():
            all_boxes.extend(v)
        all_boxes = set(all_boxes)

        for box in all_boxes:
            x1, y1, x2, y2 = GroundedCaptionDataset.parse_box_str(box)
            x1 = int(x1 / width * scale)
            y1 = int(y1 / height * scale)
            x2 = int(x2 / width * scale)
            y2 = int(y2 / height * scale)
            grounded_caption = grounded_caption.replace(box, f'({x1:03d},{y1:03d})({x2:03d},{y2:03d})')

        return grounded_caption


class DatasetWrapper(IterableDataset):
    def __init__(
        self,
        dataset: GroundingBaseDataset,
        concat_mode: bool = False,
        max_len: int = 2048,
        per_device_batch_size: int = 1,
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_name = getattr(dataset, 'dataset_name', dataset.__class__.__name__)
        # self.collator = dataset.collator
        self.collator = None
        self.concat_mode = concat_mode
        self.max_len = max_len if concat_mode else 0
        self.per_device_batch_size = per_device_batch_size

        self.epoch = 0
        self.tokenizer = dataset.collator.tokenizer

    @staticmethod
    def merge_cache(cache):
        merged_data = {}
        for key in cache[0]:
            merged_data[key] = cache[0][key]
            
        merged_data.pop('ignore_prompt_token_offset')
        merged_data.pop('meta')

        for data in cache[1:]:
            merged_data['image_tensors'] = torch.cat([merged_data['image_tensors'], data['image_tensors']], dim=0)
            merged_data['num_image_per_seq'] = merged_data['num_image_per_seq'] + data['num_image_per_seq']
            merged_data['text_ids'] = torch.cat([merged_data['text_ids'], data['text_ids']], dim=1)
            merged_data['attention_mask'] = torch.cat([merged_data['attention_mask'], data['attention_mask']], dim=1)
            merged_data['gt_text_ids'] = torch.cat([merged_data['gt_text_ids'], data['gt_text_ids']], dim=1)
        
        merged_data['concat_mode'] = True
        return merged_data

    def __iter__(self):
        assert self.dataset.collator is not None
        self.dataset.shuffle()
        
        cache = []
        yield_data = []
        cum_seq_len = 0
        for data in self.dataset:
            inputs = self.dataset.collator([data])

            assert inputs['text_ids'].shape[0] == 1
            cum_seq_len += inputs['text_ids'].shape[1]

            if cum_seq_len > self.max_len and len(cache) > 0:
                yield_data.append(DatasetWrapper.merge_cache(cache))
                cache = [inputs]
                cum_seq_len = inputs['text_ids'].shape[1]
            else:
                cache.append(inputs)

            if len(yield_data) >= self.per_device_batch_size:
                yield self.dataset.collator(yield_data)
                yield_data = []

        if len(cache) > 0:
            yield_data.append(DatasetWrapper.merge_cache(cache))

        if len(yield_data) >= self.per_device_batch_size:
            yield self.dataset.collator(yield_data)

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer


class IterableKosmos2Dataset(IterableBaseDataset):
    def __init__(
        self,
        data_root: str,
        annt_root: str,
        answer_key: str,
        query_key: Optional[str] = None,
        confidence_threshold: float = 0,
        start_idx: int = 0,
        end_idx: int = 1,
        filename_template: str = 'train_grounding_{i}.jsonl',
        transform: Optional[T.Compose] = None,
        dataset_len: Optional[int] = None,
        box_scale: int = 999,
        collator: Optional[GroundingCollator] = None,
        collate_mode: str = 'generate_grounding',
        return_image: bool = True,
        distributed: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ann = []
        self.box_scale = box_scale
        self.transform = transform
        self.collator = collator
        self.collate_mode = collate_mode
        self.return_image = return_image

        self.data_root = data_root
        self.annt_root = annt_root
        self.query_key = query_key
        self.answer_key = answer_key
        self.confidence_threshold = confidence_threshold
        self.distributed = distributed

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.dataset_len = dataset_len // self.world_size if distributed else dataset_len

        self.start_idx = start_idx
        self.end_idx = end_idx
        self.filename_template = filename_template

    def __len__(self):
        return self.dataset_len

    def __iter__(self):
        for i in range(self.start_idx, self.end_idx):
            with open(os.path.join(self.annt_root, self.filename_template.format(i=i)), 'r') as file:
                lines = file.readlines()

            for line_idx, line in enumerate(lines):
                
                if self.distributed and line_idx % self.world_size != self.rank:
                    continue
                
                ann = json.loads(line)
                
                if ann['confidence'] < self.confidence_threshold:
                    continue
                
                data = {
                    'id': -1,
                    'image': os.path.join(self.data_root, ann['image']),
                    'label': ann[self.answer_key],
                    'bbox': ann['bbox'],
                }

                image = data['image']
                data['image'] = self.loader(image).convert('RGB') if self.return_image else image

                if self.transform is not None and self.return_image:
                    data['images_tensor'] = self.transform(data['image'])

                x1, y1, x2, y2 = ann['bbox']
                assert x1 <= x2 and y1 <= y2

                data['bbox'] = (
                    int(x1 / data['image'].width * self.box_scale),
                    int(y1 / data['image'].height * self.box_scale),
                    int(x2 / data['image'].width * self.box_scale),
                    int(y2 / data['image'].height * self.box_scale),
                )

                yield data

    def shuffle(self):
        pass


class RandomResizedCrop(RandomResizedCropAndInterpolation):
    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        x1, y1, x2, y2 = bbox

        i = min(y1, i)
        j = min(x1, j)
        h = max(y2, i+h) - i
        w = max(x2, j+w) - j
        
        bbox = [x1-j, y1-i, x2-j, y2-i]
        bbox[0] = bbox[0] / w * self.size[0]
        bbox[1] = bbox[1] / h * self.size[1]
        bbox[2] = bbox[2] / w * self.size[0]
        bbox[3] = bbox[3] / h * self.size[1]
        
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation), tuple(bbox)
