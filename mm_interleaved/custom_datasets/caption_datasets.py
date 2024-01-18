import os
import json
import random

from .loader import BaseDataset


class NoCapsDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_file,
        transform,
        image_only=False,
        total_length=None,
        collate_mode='generate_texts',
        add_eos=None,
    ) -> None:
        super().__init__()
        self.collate_mode = collate_mode
        self.transform = transform
        self.data_root = data_root
        self.image_only = image_only
        self.annts = self.load_annotations(annt_file)
        self.annt_file = annt_file
        if self.image_only:
            self.dedeup_image()
        if total_length is not None:
            self.annts = self.annts[:total_length]
        self.add_eos = add_eos
        print(f"length of the dataset is {len(self.annts)}")

    def load_annotations(self, annt_file):
        meta_info = json.load(open(annt_file, "r"))
        images = meta_info['images']
        annotations = meta_info['annotations']

        image_info = {}
        for image in images:
            image_info[image['id']] = image

        processed_annotations = []
        for ann in annotations:
            image_id = ann['image_id']
            file_name = image_info[image_id]['file_name']
            caption = ann['caption']

            processed_annotations.append({
                'image': file_name,
                'caption': caption,
                'image_id': image_id,
            })

        return processed_annotations

    def dedeup_image(self):
        annts = {}
        for annt in self.annts:
            image_idx = annt["image_id"]
            if image_idx in annts:
                continue
            annts[image_idx] = annt
        self.annts = list(annts.values())

    def __repr__(self) -> str:
        return "Nocaps Dataset"

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):
        item = self.annts[index]
        caption = item["caption"]
        if isinstance(caption, list):  # TODO, random choose one caption from the image
            caption = random.choice(caption)
        caption = caption.lower()
        if self.add_eos is not None:
            caption = caption + self.add_eos
        image_idx_int = item["image_id"]
        image_path = os.path.join(self.data_root, item["image"])

        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return image, caption, image_idx_int


class Flickr30KDataset(NoCapsDataset):
    def __repr__(self) -> str:
        return "Flickr30K Dataset"
