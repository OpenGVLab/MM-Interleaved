import os.path as osp
import json
import random

from .loader import BaseDataset
from .wds_utils import init_tokenizer


class VisDialDenseDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_root,
        transform,
        tokenizer_path,
        total_length=None,
        num_img_token=32,
        collate_mode='generate_scores',
        phase="val",
    ) -> None:
        '''
            VisDial dataset only for NDCG evaluation
        '''
        super().__init__()

        assert phase == 'val'

        self.phase = phase
        self.transform = transform
        self.data_root = data_root
        self.annt_root = annt_root
        self.tokenizer = init_tokenizer(tokenizer_path)
        self.num_img_token = num_img_token
        self.collate_mode = collate_mode

        dialog_json_path = osp.join(self.annt_root, 'visdial_1.0_val.json')
        with open(dialog_json_path, 'r') as rf:
            data = json.load(rf)["data"]
        
        self.dialogs = data["dialogs"]
        self.questions = data["questions"]
        self.answers = data["answers"]

        dense_annt_path = osp.join(self.annt_root, 'visdial_1.0_val_dense_annotations.json')
        with open(dense_annt_path, 'r') as rf:
            data_dense = json.load(rf)
        self.dense_annt = {d["image_id"]:d for d in data_dense}

        if total_length is not None:
            self.dialogs = self.dialogs[:total_length]
        print(f"length of the dataset is {len(self.dialogs)}")

    def __repr__(self) -> str:
        return (
            f"VisDial Dataset phase={self.phase}\n"
            f"annotation_root={self.annt_root} data_root={self.data_root}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index):

        item = self.dialogs[index]

        image_id = item["image_id"]
        image_path = osp.join(self.data_root, "VisualDialog_val2018", f"VisualDialog_val2018_{image_id:012d}.jpg")

        try:
            image = self.loader(image_path).convert("RGB")

            image = self.transform(image)
        except:
            print(image_path)
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)
        
        image_prompt = "<|beginofimage|>" + "<|image|>" * self.num_img_token
        text = f"{image_prompt} caption: {item['caption']}. "
        dense_annt = self.dense_annt[image_id]
        round_idx = dense_annt["round_id"] - 1
        dialog = item["dialog"]
        for rnd in range(round_idx-1):
            question = self.questions[dialog[rnd]["question"]]
            answer = self.answers[dialog[rnd]["answer"]]
            text += f"question: {question}? answer: {answer}. "
        
        question = self.questions[dialog[round_idx]["question"]]
        text += f"question: {question}? answer:"

        options = dialog[round_idx]["answer_options"]
        options = [self.answers[i] for i in options]
        # gt_relevance = dense_annt["gt_relevance"]

        # assert len(gt_relevance) == len(options)

        text_tensor = self.tokenizer(
            [text],
            truncation=False,
            padding=False,
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_ids = text_tensor.input_ids[0]
        attn_mask = text_tensor.attention_mask[0]

        options_tensor = self.tokenizer(
            options,
            truncation=False,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        options_ids = options_tensor.input_ids
        options_attn_mask = options_tensor.attention_mask

        return dict(
            image_id=image_id,
            image_tensor=image,
            # context=text,
            # options=options,
            text_ids=text_ids,
            attn_mask=attn_mask,
            options_ids=options_ids[:,1:], # no <bos>
            options_attn_mask=options_attn_mask[:,1:],
            # gt_relevance=gt_relevance,
        )
