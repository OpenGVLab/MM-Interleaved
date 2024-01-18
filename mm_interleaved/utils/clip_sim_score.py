import json
from PIL import Image
from transformers import CLIPModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from ..utils.misc import MetricLogger, barrier, get_world_size, get_rank
from ..custom_datasets.clip_itp import CLIPImagePairDataset


def tensor_to_pil(images: torch.Tensor):
    pil_images = images.mul(255).add_(0.5).clamp_(0, 255)
    pil_images = [
        Image.fromarray(img.permute(1, 2, 0).to("cpu", torch.uint8).numpy()).convert("RGB")
        for img in pil_images
    ]
    return pil_images


@torch.no_grad()
def calculate_clip_sim_i2i(
    image_list,
    model_name="./assets/openai/clip-vit-large-patch14",
    device="cuda",
    batch_size=2048,
):
    if isinstance(image_list, str):
        image_list = json.load(open(image_list), "r")
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_model.to(device)
    clip_model.eval()

    clip_dataset = CLIPImagePairDataset(image_list, model_name)
    print(clip_dataset)
    num_tasks = get_world_size()
    global_rank = get_rank()
    sampler = DistributedSampler(
        clip_dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=False,
    )
    mini_batch_size = batch_size // num_tasks
    data_loader = DataLoader(
        clip_dataset,
        sampler=sampler,
        batch_size=mini_batch_size,
        drop_last=False,
        num_workers=10,
        pin_memory=True,
    )

    metric_logger = MetricLogger(delimiter="  ")
    header = "Eval CLIP similarity i2i: "
    print_freq = 5

    for batch_idx, data in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        image, image_gt, image_idx = data
        image = image.to(device, non_blocking=True)
        image_gt = image_gt.to(device, non_blocking=True)

        image_feat = clip_model.get_image_features(pixel_values=image)
        image_gt_feat = clip_model.get_image_features(pixel_values=image_gt)

        # Compute cosine similarity.
        image_feat = F.normalize(image_feat, dim=-1)
        image_gt_feat = F.normalize(image_gt_feat, dim=-1)
        scores = (image_feat * image_gt_feat).sum(dim=-1)
        metric_logger.meters["clip_sim_i2i"].update(scores.mean(), n=scores.shape[0])

    barrier()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    score = metric_logger.meters["clip_sim_i2i"].global_avg
    print("CLIP similarity:", score)
    return score


@torch.no_grad()
def clip_rerank_generated_images(
    images: torch.Tensor,
    captions,
    clip_model,
    clip_processor,
    device="cuda",
):
    _images = tensor_to_pil(images)
    images = _images

    bs = len(captions)
    num_candidates = len(images) // len(captions)

    data = clip_processor(
        images=_images,
        text=captions,
        return_tensors="pt",
        padding="max_length",
        max_length=77,
    )

    image_tensors = data.pixel_values.to(device=device)
    text_ids = data.input_ids.to(device=device)

    image_feat = clip_model.get_image_features(pixel_values=image_tensors)
    image_feat = F.normalize(image_feat, dim=-1)
    text_feat = clip_model.get_text_features(input_ids=text_ids)
    text_feat = F.normalize(text_feat, dim=-1)
    text_feat = text_feat.repeat(num_candidates, 1)

    scores = (image_feat * text_feat).sum(dim=-1)
    scores = scores.view(num_candidates, -1).transpose(0, 1)

    best_image_idxs = scores.argmax(dim=1)
    best_images = [images[idx * bs + i] for i,idx in enumerate(best_image_idxs)]

    return best_images
