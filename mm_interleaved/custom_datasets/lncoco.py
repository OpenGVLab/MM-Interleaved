import os
import json
import random
import numpy as np
from collections import Counter

from .loader import BaseDataset


class LNCOCODataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        image_only=False,
        total_length=None,
        collate_mode="generate_images",
        phase="val",
        add_eos=None,
    ) -> None:
        super().__init__()
        assert phase == "val" and collate_mode in ["generate_images"]
        self.collate_mode = collate_mode
        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root
        self.phase = phase
        self.image_only = image_only

        annt_file = os.path.join(annt_root, "coco_val_captions.jsonl")
        with open(annt_file, "r") as rf:
            data = rf.readlines()
        self.annts = [json.loads(s) for s in data]
        self.annt_file = annt_file
        if self.image_only:
            self.dedeup_image()
        if total_length is not None:
            if total_length <= len(self.annts):
                self.annts = self.annts[:total_length]
            else:
                # over sampling
                cnter_image = Counter([a["image_id"] for a in self.annts])
                annts_weight = [1./cnter_image[a["image_id"]] for a in self.annts]
                annts_weight = [w / sum(annts_weight) for w in annts_weight]
                annts_n = np.random.choice(self.annts, total_length - len(self.annts), p=annts_weight)
                self.annts += list(annts_n)
        self.add_eos = add_eos
        print(f"length of the dataset is {len(self.annts)}")

    def dedeup_image(self):
        annts = {}
        for annt in self.annts:
            image_idx = annt["image_id"]
            if image_idx in annts:
                continue
            annts[image_idx] = annt
        self.annts = list(annts.values())

    def image_id_to_path(self, image_id):
        # coco-2017
        return os.path.join(self.data_root, "val2017", f"{image_id:012d}.jpg")

    def __repr__(self) -> str:
        return (
            f"LNCOCO Dataset phase={self.phase}\n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):
        item = self.annts[index]
        caption = item["caption"]
        # caption = caption.lower()
        if self.add_eos is not None:
            caption = caption + self.add_eos

        image_idx_int = int(item["image_id"])
        image_path = os.path.join(self.data_root, "val2017", f"{image_idx_int:012d}.jpg")

        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return image, caption, image_idx_int
