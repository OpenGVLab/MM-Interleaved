import io
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset, IterableDataset

import logging
import os

LOG_LOADER = os.environ.get("LOG_LOADER", False)


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    return Image.open(buff)


def cv2_loader(img_bytes):
    # assert(img_bytes is not None)
    img_mem_view = memoryview(img_bytes)
    img_array = np.frombuffer(img_mem_view, np.uint8)
    imgcv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    imgcv2 = cv2.cvtColor(imgcv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(imgcv2)


class LocalClient():
    def __init__(self, **kwargs) -> None:
        pass

    def get(self, url):
        with open(url, "rb") as rf:
            data = rf.read()
        return data


class BaseLoader(object):
    def __init__(self):
        self.client = LocalClient()

    def __call__(self, fn):
        try:
            if self.client is not None:
                img_value_str = self.client.get(fn)
                img = pil_loader(img_value_str)
            else:
                img = Image.open(fn)
        except:
            try:
                img = cv2_loader(img_value_str)
            except Exception as exn:
                exn.args = exn.args + (fn,)
                if LOG_LOADER:
                    logging.warning(f"Handling BaseLoader image reading error ({repr(exn)}). Ignoring.")
                # print('Read image failed ({})'.format(fn))
                return None
            else:
                return img
        else:
            return img


class BaseDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.loader = BaseLoader()
        self.client = self.loader.client

    def __getitem__(self, index):
        raise NotImplementedError


class IterableBaseDataset(IterableDataset):
    def __init__(self) -> None:
        super().__init__()
        self.loader = BaseLoader()
        self.client = self.loader.client

    def __iter__(self):
        raise NotImplementedError

