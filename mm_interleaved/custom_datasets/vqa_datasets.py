import os
import json
from .loader import BaseDataset


class VQABaseDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        annt_file,
        transform=None,
        total_length=None,
        phase='train',
        collate_mode='generate_vqa',
        add_eos=None,
    ):
        super().__init__()
        self.collate_mode = collate_mode
        self.transform = transform
        self.data_root = data_root
        self.annt_file = annt_file
        self.phase = phase
        if total_length is not None:
            self.annts = self.annts[:total_length]
        self.add_eos = add_eos
        self.ann = self.load_annotations()
        print(f"length of the {self.__class__.__name__} is {len(self.ann)}")

    def load_annotations(self):
        raise NotImplementedError

    def __getitem__(self, index):
        ann = self.ann[index]
        image = self.loader(os.path.join(self.data_root, ann['file_name'])).convert('RGB')
        image = self.transform(image) if self.transform is not None else image
        question = ann['question']
        answer = ann['answer']
        question_id = ann.get('question_id', -1)

        return image, question, answer, question_id

    def __len__(self):
        return len(self.ann)

    @property
    def data_shape(self):
        return 4, 32, 32


class VQAV2Dataset(VQABaseDataset):
    def __init__(
        self,
        data_root='./assets/coco/images',
        annt_root='./assets/VQAv2',
        phase='train',
        ann_name_format='v2_mscoco_{split}2014_annotations.json',
        question_name_format='v2_OpenEnded_mscoco_{split}2014_questions.json',
        **kwargs,
    ):
        self.question_file = os.path.join(annt_root, question_name_format.format(split=phase))

        data_root = os.path.join(data_root, f'{phase}2014')
        annt_file = os.path.join(annt_root, ann_name_format.format(split=phase))
        super().__init__(data_root=data_root, annt_file=annt_file, phase=phase, **kwargs)

    def load_annotations(self):
        answers_info = json.load(open(self.annt_file))['annotations']
        questions_info = json.load(open(self.question_file))['questions']

        annotations = {}
        for info in answers_info:
            image_id = info['image_id']
            question_id = info['question_id']
            answer = info['multiple_choice_answer'] if 'multiple_choice_answer' in info else info['answers'][0]['answer']

            assert question_id not in annotations
            annotations[question_id] = {
                'image_id': image_id,
                'question_id': question_id,
                'answer': answer,
                'file_name': f'COCO_{self.phase}2014_{image_id:012d}.jpg',
            }

        for info in questions_info:
            image_id = info['image_id']
            question_id = info['question_id']
            question = info['question']

            assert annotations[question_id]['image_id'] == image_id
            annotations[question_id]['question'] = question

        return list(annotations.values())


class OKVQADataset(VQAV2Dataset):
    def __init__(
        self,
        annt_root='./assets/OK-VQA',
        ann_name_format='mscoco_{split}2014_annotations.json',
        question_name_format='OpenEnded_mscoco_{split}2014_questions.json',
        **kwargs,
    ):
        super().__init__(annt_root=annt_root, ann_name_format=ann_name_format, question_name_format=question_name_format, **kwargs)


class VizWizVQADataset(VQABaseDataset):
    def __init__(
        self,
        data_root='./assets/VizWiz',
        annt_root='./assets/VizWiz-VQA',
        phase='train',
        batch_size=4,
        **kwargs,
    ):
        data_root = os.path.join(data_root, phase)
        annt_file = os.path.join(annt_root, f'{phase}.json')
        super().__init__(data_root=data_root, annt_file=annt_file, phase=phase, **kwargs)
        self.batch_size = batch_size

    def load_annotations(self):
        meta_info = json.load(open(self.annt_file))

        annotations = []
        for ann in meta_info:
            annotations.append({
                'question_id': int(ann['image'].split('_')[-1].split('.')[0]),
                'file_name': ann['image'],
                'question': ann['question'],
                'answer': ann['answers'][0]['answer'],
            })

        return annotations

class TextVQADataset(VQABaseDataset):
    def __init__(
        self,
        data_root='./assets/TextVQA/train_images',
        annt_root='./assets/TextVQA',
        phase='train',
        ann_name_format='textvqa_{split}_annotations.json',
        question_name_format='textvqa_{split}_questions.json',
        **kwargs,
    ):
        self.question_file = os.path.join(annt_root, question_name_format.format(split=phase))

        annt_file = os.path.join(annt_root, ann_name_format.format(split=phase))
        super().__init__(data_root=data_root, annt_file=annt_file, phase=phase, **kwargs)

    def load_annotations(self):
        answers_info = json.load(open(self.annt_file))['annotations']
        questions_info = json.load(open(self.question_file))['questions']

        annotations = {}
        for info in answers_info:
            image_id = info['image_id']
            question_id = info['question_id']
            answer = info['multiple_choice_answer'] if 'multiple_choice_answer' in info else info['answers'][0]['answer']

            assert question_id not in annotations
            annotations[question_id] = {
                'image_id': image_id,
                'question_id': question_id,
                'answer': answer,
            }

        for info in questions_info:
            image = info['image']
            image_id = info['image_id']
            question_id = info['question_id']
            question = info['question']

            assert annotations[question_id]['image_id'] == image_id
            annotations[question_id]['question'] = question
            annotations[question_id]['file_name'] = image

        return list(annotations.values())
