import os
import json
import random

from .loader import BaseDataset


class CocoCaptionKarpathyDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        image_only=False,
        total_length=None,
        collate_mode="generate_texts",
        phase="train",
        year="2014",
        add_eos=None,
        use_1st_sentence_only=True,
        rerank_by_clip=False,
    ) -> None:
        super().__init__()
        self.collate_mode = collate_mode
        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root
        self.phase = phase
        self.year = year
        self.image_only = image_only
        annt_file = os.path.join(
            annt_root, "annotations", f"coco_karpathy_{phase}.json"
        )
        self.annts = json.load(open(annt_file, "r"))
        self.annt_file = annt_file
        if self.image_only:
            self.dedeup_image()
        if total_length is not None:
            self.annts = self.annts[:total_length]
        self.add_eos = add_eos
        self.use_1st_sentence_only = use_1st_sentence_only
        self.rerank_by_clip = rerank_by_clip
        print(f"length of the dataset is {len(self.annts)}")

    def dedeup_image(self):
        annts = {}
        for annt in self.annts:
            image_idx = annt["image"].split("_")[-1][
                :-4
            ]  # 'val2014/COCO_val2014_000000391895.jpg'
            if image_idx in annts:
                continue
            annts[image_idx] = annt
        self.annts = list(annts.values())

    def image_id_to_path(self, image_id):
        phase = "val" if self.phase == "test" else self.phase
        # coco-2014
        image_idx = str(image_id).zfill(12)
        image_name = f"COCO_{phase}{self.year}_{image_idx}.jpg"
        image_path = os.path.join(
            self.data_root, f"{phase}{self.year}", image_name
        )
        return image_path

    def __repr__(self) -> str:
        return (
            f"MSCOCO-Caption Karpathy Dataset year={self.year} phase={self.phase}\n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def __getitem__(self, index):
        item = self.annts[index]
        caption = item["caption"]
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = caption.lower()
        if self.add_eos is not None:
            caption = caption + self.add_eos
        image_idx_int = int(item["image"].split("_")[-1][:-4])
        image_name = item["image"]
        image_path = os.path.join(self.data_root, f"{image_name}")

        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return image, caption, image_idx_int
