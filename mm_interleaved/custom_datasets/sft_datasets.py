import json
import os
import random

import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from .loader import BaseDataset


class LLaVADataset(BaseDataset):
    def __init__(
            self,
            annt_root=[],
            data_root=[],
            transform=None,
    ):
        super().__init__()
        self.ann_path = [annt_root] if isinstance(annt_root, str) else annt_root
        self.data_root = [data_root] if isinstance(data_root, str) else data_root
        self.transform = transform
        
        self.ann = []
        print("Formatting inputs...Skip in lazy mode")
        for index, p in enumerate(self.ann_path):
            if p.endswith('json'):
                with open(p, 'r') as file:
                    data = json.load(file)
                    for item in data:
                        try:
                            item['image'] = os.path.join(self.data_root[index], item['image'])
                            self.ann.append(item)
                        except:
                            pass
            elif p.endswith('.jsonl'):
                for line in open(p, 'r'):
                    data = json.loads(line)
                    try:
                        data['image'] = os.path.join(self.data_root[index], data['image'])
                        self.ann.append(data)
                    except:
                        pass
            
        # split multi-round dialogues to single-round dialogue
        max_conv_num = 2  # 1 round
        print(f"data length before split: {len(self.ann)}")
        new_ann = []
        for item in self.ann:
            conversations = item["conversations"]
            conversations = [conversations[i:i + max_conv_num] for i in range(0, len(conversations), max_conv_num)]
            for conv in conversations:
                new_item = item.copy()
                if "<image>" not in conv[0]['value']:
                    conv[0]['value'] = "<image>\n" + conv[0]['value']
                new_item["conversations"] = conv
                new_ann.append(new_item)
        self.ann = new_ann
        print(f"data length after split: {len(self.ann)}")
    
    def __getitem__(self, index):
        while True:
            try:
                data = self.ann[index]
                
                assert len(data['conversations']) == 2
                
                query = data['conversations'][0]['value'].replace('<image>\n', '')
                query = query.replace('\n<image>', '')
                query = query.replace('<image>', '')
                
                image_id = data['id']
                image = self.loader(data['image']).convert('RGB')
                label = data['conversations'][1]['value']
                break
            except Exception as e:
                print(e)
                print('Error loading data:', data['image'])
                index = random.randint(0, len(self.ann) - 1)
        
        return self.transform(image), query, label, image_id
    
    def __len__(self):
        return len(self.ann)


class WeightedConcatDataset(ConcatDataset):
    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = torch.DoubleTensor(weights)
        self.total_size = sum(len(d) for d in datasets)
        self.sampler = WeightedRandomSampler(weights=self.weights, num_samples=self.total_size, replacement=True)
    
    def __iter__(self):
        return iter(self.sampler)
    
    def __len__(self):
        return self.total_size
