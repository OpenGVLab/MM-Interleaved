import os
import json
import random
import numpy as np

from .loader import BaseDataset


class CocoCaptionDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        image_only=False,
        total_length=None,
        collate_mode="generate_images",
        shuffle=False,
        rerank_by_clip=False,
        phase="train",
        year="2014",
    ) -> None:
        super().__init__()
        self.collate_mode = collate_mode
        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root
        self.phase = phase
        self.year = year
        self.image_only = image_only
        self.rerank_by_clip = rerank_by_clip

        annt_file = os.path.join(
            annt_root, "annotations", f"captions_{phase}{year}.json"
        )
        self.annt_file = annt_file
        self.annts = json.load(open(annt_file, "r"))["annotations"]
        if self.image_only:
            self.dedeup_image()
        if shuffle:
            np.random.shuffle(self.annts)
        if total_length is not None:
            self.annts = self.annts[:total_length]
        print(f"length of the dataset is {len(self.annts)}")

    def dedeup_image(self):
        annts = {}
        for annt in self.annts:
            image_idx = str(annt["image_id"]).zfill(12)
            if image_idx in annts:
                continue
            annts[image_idx] = annt
        self.annts = list(annts.values())

    def image_id_to_path(self, image_id):
        # coco-2014
        image_idx = str(image_id).zfill(12)
        image_name = f"COCO_{self.phase}{self.year}_{image_idx}.jpg"
        image_path = os.path.join(
            self.data_root, f"{self.phase}{self.year}", image_name
        )
        return image_path

    def __repr__(self) -> str:
        return (
            f"MSCOCO-Caption Dataset year={self.year} phase={self.phase}\n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):
        item = self.annts[index]
        caption = item["caption"].lower()

        image_idx = str(item["image_id"]).zfill(12)
        image_name = f"COCO_{self.phase}{self.year}_{image_idx}.jpg"
        image_path = os.path.join(
            self.data_root, f"{self.phase}{self.year}", image_name
        )
        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return image, caption, item["image_id"]
