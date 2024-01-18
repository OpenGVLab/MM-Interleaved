import os
import json
import random

from .loader import BaseDataset


class Image2ParagraphDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        image_only=False,
        total_length=None,
        collate_mode="generate_texts",
        phase="train",
        add_eos=None,
    ) -> None:
        super().__init__()
        self.collate_mode = collate_mode
        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root
        self.phase = phase
        self.image_only = image_only

        annt_file = os.path.join(annt_root, "annotations", f"paragraphs_coco.json")
        with open(annt_file, "r") as rf:
            data = json.load(rf)
        annts = {d["image_id"]: d for d in data["annotations"]}

        split_file = os.path.join(annt_root, "annotations", f"{phase}_split.json")
        with open(split_file, "r") as rf:
            split_idxs = set(json.load(rf))
        annts = [v for k, v in annts.items() if k in split_idxs]

        self.annts = annts
        self.annt_file = annt_file
        if total_length is not None:
            self.annts = self.annts[:total_length]
        self.add_eos = add_eos
        print(f"length of the dataset is {len(self.annts)}")

    def __repr__(self) -> str:
        return (
            f"Image2Paragraph Dataset phase={self.phase}\n"
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

        image_idx_int = item["image_id"]
        image_subpaths = item["url"].split("/")[-2:]
        image_path = os.path.join(self.data_root, *image_subpaths)

        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        return image, caption, image_idx_int
