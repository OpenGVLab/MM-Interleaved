import os
from transformers import CLIPProcessor

from .loader import BaseDataset


class CLIPImageTextPairDataset(BaseDataset):
    def __init__(
        self,
        image_root,
        caption_list,
        model_name="openai/clip-vit-large-patch14",
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.image_root = image_root
        self.caption_list = caption_list

        self.clip_processor = CLIPProcessor.from_pretrained(model_name)

        print(f"length of the dataset is {len(self.caption_list)}")

    def __repr__(self) -> str:
        return (
            f"CLIPImageTextPair Dataset total_length={len(self)}\n"
            f"image_root={self.image_root}\nprocessor={self.clip_processor}"
        )

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, index):
        caption = self.caption_list[str(index)]["caption"]
        image_path = os.path.join(self.image_root, f"{index:05d}.png")

        image = self.loader(image_path).convert("RGB")
        data = self.clip_processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
        )

        return data.pixel_values[0], data.input_ids[0], index


class CLIPImagePairDataset(BaseDataset):
    def __init__(
        self,
        image_pair_list,
        model_name="openai/clip-vit-large-patch14",
    ) -> None:

        super().__init__()

        self.model_name = model_name
        self.image_pair_list = image_pair_list

        self.clip_processor = CLIPProcessor.from_pretrained(model_name)

        print(f"length of the dataset is {len(self.image_pair_list)}")

    def __repr__(self) -> str:
        return (
            f"CLIPImagePairDataset total_length={len(self)}\n"
            f"processor={self.clip_processor}"
        )

    def __len__(self):
        return len(self.image_pair_list)

    def __getitem__(self, index):
        image_path = self.image_pair_list[index]["image_path"]
        image = self.loader(image_path).convert("RGB")

        image = self.clip_processor(
            images=image,
            text=None,
            return_tensors="pt",
        ).pixel_values[0]

        image_path_gt = self.image_pair_list[index]["image_gt_path"]
        image_gt = self.loader(image_path_gt).convert("RGB")

        image_gt = self.clip_processor(
            images=image_gt,
            text=None,
            return_tensors="pt",
        ).pixel_values[0]

        return image, image_gt, index
