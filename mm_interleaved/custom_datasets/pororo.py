from PIL import Image
import os
import json
import random
import numpy as np

from .loader import BaseDataset


class PororoDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        phase="train",
        collate_mode="train",
        add_eos="",
        num_img_token=32,
        img_first_prob=0.0,
        add_soi_token=True,
        context_type="multi_modal",
        target_image_idxs=None,
    ):
        super().__init__()

        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root

        assert phase in ["train", "val", "test"]
        self.phase = phase

        assert collate_mode in ["train", "generate_images"]
        self.collate_mode = collate_mode
        self.add_eos = add_eos

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

        self.target_image_idxs = target_image_idxs
        self.save_gt_image_online = True
        self.load_database()
        print(f"length of the dataset is {len(self.annts)}")

    def load_database(self):
        self.main_characters = [
            "Pororo",
            "Loopy",
            "Eddy",
            "Harry",
            "Poby",
            "Tongtong",
            "Crong",
            "Rody",
            "Petty",
        ]

        self.descriptions = np.load(
            os.path.join(self.annt_root, "descriptions.npy"),
            allow_pickle=True,
            encoding="latin1",
        ).item()

        self.imgs_list = np.load(
            os.path.join(self.annt_root, "img_cache4.npy"), encoding="latin1"
        )
        self.followings_list = np.load(
            os.path.join(self.annt_root, "following_cache4.npy")
        )

        phase_to_ids = {
            "train": 0,
            "val": 1,
            "test": 2,
        }

        annt_ids = np.load(
            os.path.join(self.annt_root, "train_seen_unseen_ids.npy"), allow_pickle=True
        )
        annt_ids = annt_ids[phase_to_ids[self.phase]]
        self.annts = np.sort(annt_ids)

    def __repr__(self) -> str:
        return (
            f"Pororo Dataset phase={self.phase}\n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def _get_caption(self, global_id):
        caption = self.descriptions[str(global_id).replace(".png", "")][0]
        caption = caption.lower()
        for char in self.main_characters:
            if char.lower() in caption:
                caption = caption.replace(char.lower(), char)
        caption = caption.replace("\n", "").replace("\t", "").strip()

        return caption

    def meta_to_image(self, meta, target_image_idx=-1):
        item_id, image_frame_idxs = meta
        item_id = int(item_id)
        global_ids = [str(self.imgs_list[item_id])[2:-1]] + [
            str(self.followings_list[item_id][i])[2:-1] for i in range(4)
        ]

        global_id = global_ids[target_image_idx]
        frame_idx = image_frame_idxs[target_image_idx]
        image_path = os.path.join(self.data_root, global_id)
        image = self.loader(image_path).convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = image[frame_idx * 128 : (frame_idx + 1) * 128]
        image = Image.fromarray(image, "RGB").convert("RGB")

        return image

    def _get_global_ids(self, item_id):
        item_id = int(item_id)
        global_ids = [str(self.imgs_list[item_id])[2:-1]] + [
            str(self.followings_list[item_id][i])[2:-1] for i in range(4)
        ]

        return global_ids

    def _get_image(
        self, global_id=None, return_frame_idx=False, frame_idx=-1, image_path=None
    ):
        try:
            if image_path is None:
                image_path = os.path.join(self.data_root, global_id)
            image = self.loader(image_path).convert("RGB")

            # random sample 1 single frame of shape [128, 128, 3]
            image = np.array(image).astype(np.uint8)
            if frame_idx < 0:
                frame_idx = random.randint(0, image.shape[0] / 128 - 1)
            image = image[frame_idx * 128 : (frame_idx + 1) * 128]
            image = Image.fromarray(image, "RGB").convert("RGB")

            if self.transform is not None:
                image = self.transform(image)
        except Exception as e:
            print(e)
            print(image_path)
            image = None

        if return_frame_idx:
            return image, frame_idx
        return image

    def _get_item_id_image_paths_captions(self, index=None, item_id=None):
        if item_id is None:
            item_id = self.annts[index]

        global_ids = self._get_global_ids(item_id)

        image_paths = [
            os.path.join(self.data_root, global_id) for global_id in global_ids
        ]

        captions = [self._get_caption(global_id) for global_id in global_ids]

        return item_id, image_paths, captions

    def __getitem__(self, index):
        item_id, image_paths, captions = self._get_item_id_image_paths_captions(index)
        meta = [str(item_id)]

        images_tensor = []
        text = ""

        if self.collate_mode == "train":
            assert self.phase == "train"

            for i in range(len(image_paths)):
                image = self._get_image(image_path=image_paths[i])
                caption = captions[i]

                if np.random.random() < self.img_first_prob:
                    _text = f"{self.image_subseq}{caption} "
                else:
                    _text = f"{caption}{self.image_subseq} "

                text += _text
                images_tensor.append(image)

        else:
            assert self.phase != "train"
            image_frame_idxs = []

            # prepare history context
            if self.context_type == "multi_modal":
                for i in range(len(image_paths) - 1):
                    image, image_frame_idx = self._get_image(
                        image_path=image_paths[i],
                        return_frame_idx=True,
                    )
                    image_frame_idxs.append(image_frame_idx)
                    caption = captions[i]

                    if np.random.random() < self.img_first_prob:
                        _text = f"{self.image_subseq}{caption} "
                    else:
                        _text = f"{caption}{self.image_subseq} "

                    text += _text
                    images_tensor.append(image)

            elif self.context_type == "image_only":
                for i in range(len(image_paths) - 1):
                    image, image_frame_idx = self._get_image(
                        image_path=image_paths[i],
                        return_frame_idx=True,
                    )
                    image_frame_idxs.append(image_frame_idx)
                    text += self.image_subseq
                    images_tensor.append(image)

            elif self.context_type == "text_only":
                for i in range(len(image_paths) - 1):
                    caption = captions[i]
                    text += f"{caption} "

            # prepare target
            if self.collate_mode == "generate_images":
                caption = captions[-1]
                if self.context_type != "image_only":
                    text += caption

                image, image_frame_idx = self._get_image(
                    image_path=image_paths[-1],
                    return_frame_idx=True,
                )
                image_frame_idxs.append(image_frame_idx)
                text += self.image_subseq
                images_tensor.append(image)

            meta.append(image_frame_idxs)

        text = text.strip()
        if self.add_eos:
            text += self.add_eos

        return dict(text=text, images_tensor=images_tensor, meta=meta)

    @property
    def task_prefix(self):
        return f"_{self.context_type}"
