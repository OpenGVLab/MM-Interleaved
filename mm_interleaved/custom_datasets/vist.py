import os
import json
import numpy as np

from .loader import BaseDataset


class VISTDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        total_length=None,
        phase="train",
        
        collate_mode="generate_texts",
        add_eos="",
        num_img_token=32,
        img_first_prob=0.0,
        add_soi_token=True,
        round_range="last",
        context_type="current",
    ):
        super().__init__()

        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root

        assert phase in ["train", "val", "test"]
        self.phase = phase

        assert collate_mode in ["train", "generate_texts", "generate_images"]
        self.collate_mode = collate_mode
        self.add_eos = add_eos

        assert round_range in ["last", "all"]
        self.round_range = round_range

        assert context_type in [
            "multi_modal",
            "image_only",
            "text_only",
            "current",
        ]
        self.context_type = context_type

        self.num_img_token = num_img_token
        self.img_first_prob = img_first_prob
        self.add_soi_token = add_soi_token

        self.image_subseq = "<|image|>" * self.num_img_token
        if self.add_soi_token:
            self.image_subseq = "<|beginofimage|>" + self.image_subseq

        annt_file = os.path.join(
            annt_root, "annotations", f"{phase}_formatted_filtered.json"
        )
        self.annt_file = annt_file
        self.load_database()

        if total_length is not None:
            self.annts = self.annts[:total_length]

        print(f"length of the dataset is {len(self.annts)}")

    def load_database(self):
        with open(self.annt_file, "r") as rf:
            annts = json.load(rf)["annotations"]

        data = []
        for k, v in annts.items():
            v.sort(key=lambda x: x["sequence_index"])
            data.append(dict(story_id=k, story=v))
        data.sort(key=lambda x: x["story_id"])

        if self.round_range == "all":
            assert self.phase != "train"
            data_n = []
            for d in data:
                for i in range(1, len(d["story"])):
                    d_n = dict(story_id=f"{d['story_id']}_{i}", story=d["story"][:i])
                    data_n.append(d_n)
            data = data_n

        self.annts = data

    def __repr__(self) -> str:
        return (
            f"VIST Dataset phase={self.phase}\n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def _get_image(self, image_id, return_image_path=False):
        try:
            image_path = os.path.join(
                self.data_root, "images", f"{self.phase}_images", f"{image_id}.png"
            )
            image = self.loader(image_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(e)
            print(image_path)
            image = None

        if return_image_path:
            return image, image_path
        return image

    def __getitem__(self, index):
        item = self.annts[index]["story"]
        meta = [self.annts[index]["story_id"]]

        images_tensor = []
        text = ""
        if self.collate_mode == "train":
            assert self.phase == "train"

            # no target image / text
            for i in range(len(item)):
                turn = item[i]
                image = self._get_image(turn["image_id"])
                if np.random.random() < self.img_first_prob:
                    _text = f"{self.image_subseq}{turn['caption']} "
                else:
                    _text = f"{turn['caption']}{self.image_subseq} "

                text += _text
                images_tensor.append(image)

        else:
            assert self.phase != "train"

            # prepare history context
            if self.context_type == "multi_modal":
                for i in range(len(item) - 1):
                    turn = item[i]
                    image = self._get_image(turn["image_id"])
                    if np.random.random() < self.img_first_prob:
                        _text = f"{self.image_subseq}{turn['caption']} "
                    else:
                        _text = f"{turn['caption']}{self.image_subseq} "

                    text += _text
                    images_tensor.append(image)

            elif self.context_type == "image_only":
                for i in range(len(item) - 1):
                    turn = item[i]
                    image = self._get_image(turn["image_id"])
                    text += self.image_subseq
                    images_tensor.append(image)

            elif self.context_type == "text_only":
                for i in range(len(item) - 1):
                    turn = item[i]
                    text += f"{turn['caption']} "

            # prepare target
            if self.collate_mode == "generate_texts":
                turn = item[-1]

                if self.context_type != "text_only":
                    image = self._get_image(turn["image_id"])
                    text += self.image_subseq
                    images_tensor.append(image)

                meta.append(turn["caption"])

            elif self.collate_mode == "generate_images":
                turn = item[-1]
                if self.context_type != "image_only":
                    text += turn["caption"]

                image, image_path = self._get_image(
                    turn["image_id"], return_image_path=True
                )
                text += self.image_subseq
                images_tensor.append(image)

                meta.append(image_path)

        text = text.strip()
        if self.add_eos:
            text += self.add_eos

        return dict(text=text, images_tensor=images_tensor, meta=meta)

    @property
    def task_prefix(self):
        return f"_{self.context_type}_{self.round_range}"
