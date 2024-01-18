import re
import json
import torch
from torchvision.ops.boxes import box_area

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def parse_box(box_str):
    PATTERN = re.compile(r'\((.*?)\)\((.*?)\)')
    predict_bbox = re.findall(PATTERN, box_str)
    
    try:
        if ',' not in predict_bbox[0][0] or ',' not in predict_bbox[0][1]:
            predict_bbox = (0., 0., 0., 0.)
        else:
            x1, y1 = [
                float(tmp) for tmp in predict_bbox[0][0].split(',')
            ]
            x2, y2 = [
                float(tmp) for tmp in predict_bbox[0][1].split(',')
            ]
            predict_bbox = (x1, y1, x2, y2)
    except:
        predict_bbox = (0., 0., 0., 0.)
    
    return predict_bbox

def grounding_eval(results_file):
    results = json.load(open(results_file))

    total_cnt = 0
    correct = 0
    for item in results:
        gt_box = item['gt_box']
        pred_box = item['pred_box']
        h = item['height']
        w = item['width']

        pred_box = parse_box(pred_box)
        pred_box = torch.tensor(pred_box, dtype=torch.float32).view(-1, 4) / 999
        pred_box[:, 0::2] *= w
        pred_box[:, 1::2] *= h

        gt_box = torch.tensor(gt_box, dtype=torch.float32).view(-1, 4) / 999
        gt_box[:, 0::2] *= w
        gt_box[:, 1::2] *= h

        iou, _ = box_iou(pred_box, gt_box)
        iou = iou.item()
        total_cnt += 1
        if iou >= 0.5:
            correct += 1

    return {'accuracy': correct / total_cnt}
