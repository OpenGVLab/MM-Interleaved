from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import numpy as np

processor = None # OneFormerProcessor.from_pretrained("./assets/shi-labs/oneformer_ade20k_dinat_large")
model = None # OneFormerForUniversalSegmentation.from_pretrained("./assets/shi-labs/oneformer_ade20k_dinat_large")


def calculate_segm(image, gt_img):
    global processor
    global model
    if processor is None:
        processor = OneFormerProcessor.from_pretrained("./assets/shi-labs/oneformer_ade20k_dinat_large")
    if model is None:
        model = OneFormerForUniversalSegmentation.from_pretrained("./assets/shi-labs/oneformer_ade20k_dinat_large")

    semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
    semantic_outputs = model(**semantic_inputs)
    # pass through image_processor for postprocessing
    predicted_semantic_map = processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[gt_img.size[::-1]])[0]

    return predicted_semantic_map

def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # imPred += 1
    # imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


def calculate_miou_given_paths(paths, num_classes=150):

    all_intersection = None
    all_union = None

    for path1, path2 in zip(*paths):
        seg_label = np.array(Image.open(path1))
        pred = np.array(Image.open(path2)) + 1

        intersection, union = intersectionAndUnion(pred, seg_label, num_classes)
        all_intersection = intersection if all_intersection is None else all_intersection + intersection
        all_union = union if all_union is None else all_union + union

    iou = all_intersection / (all_union + 1e-10)

    miou = iou.mean()

    return miou

