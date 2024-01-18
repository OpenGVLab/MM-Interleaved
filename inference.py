import os
import os.path as osp
import json
from PIL import Image
from datetime import datetime
import numpy as np
import torch
from torchvision.utils import save_image

from mm_interleaved.models.utils.monkey_patch import (
    replace_llama_attn_with_flash_attn,
    replace_blip2_attn_with_qknorm_attn,
    replace_beam_search,
    replace_stable_diffusion_pipeline_call,
    replace_stable_diffusion_unet_forward,
)

replace_beam_search()
replace_blip2_attn_with_qknorm_attn()
replace_stable_diffusion_unet_forward()
replace_stable_diffusion_pipeline_call()
IS_TRAIN = False
if IS_TRAIN:
    replace_llama_attn_with_flash_attn()


from mm_interleaved.models import MMInterleaved
from mm_interleaved.custom_datasets.utils import create_transform
from mm_interleaved.custom_datasets.wds_utils import init_tokenizer
from mm_interleaved.utils import (
    ArgumentParser,
    TrainingArguments,
    init_distributed_mode,
    load_model_weights,
)
from mm_interleaved.utils.clip_sim_score import tensor_to_pil


def load_annt_data(
    transform,
    tokenizer,
    num_total_token=2048,
    truncation=True,
    num_img_token=64,
    generation_kwargs=None,
    annt_path="",
):
    with open(annt_path, "r") as rf:
        infos = json.load(rf)

    data = []
    for info in infos:
        sentences = info["sentences"]
        sentence_ixs = info["sentence_ixs"]
        image_paths = info["images"]
        image_first = info["image_first"]

        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image = transform(image)
            images.append(image)
        assert len(images) > 0, "Please provide at least 1 image as inputs"

        image_tensors = np.stack(images, axis=0)
        image_subseq = "<|beginofimage|>" + "<|image|>" * num_img_token
        for ix, img_first in zip(sentence_ixs, image_first):
            if img_first:
                sentences[ix] = image_subseq + sentences[ix]
            else:
                sentences[ix] = sentences[ix] + image_subseq
        text = " ".join(sentences)

        # whitespace cleanup
        text = (
            text.replace("<|image|> ", "<|image|>")
            .replace(" <|image|>", "<|image|>")
            .replace(" <|beginofimage|>", "<|beginofimage|>")
            .replace("<|beginofimage|> ", "<|beginofimage|>")
        )

        tokenizer.padding_side = "right"
        text_tensor = tokenizer(
            text,
            max_length=num_total_token,
            truncation=truncation,
            padding="do_not_pad",
            return_tensors="np",
            return_attention_mask=True,
        )
        text_ids = text_tensor["input_ids"][0]
        text_attn_mask = text_tensor["attention_mask"][0]

        image_tensors = torch.from_numpy(image_tensors)
        num_images = image_tensors.shape[0]
        target_image_idxs = torch.tensor([num_images - 1], dtype=torch.long)

        _data = dict(
            image_tensors=image_tensors,
            image_tensors_dec=None,
            text_ids=torch.from_numpy(text_ids)[None, ...],
            attention_mask=torch.from_numpy(text_attn_mask)[None, ...],
            num_image_per_seq=torch.tensor([num_images]),
            nearest_bos_idxs=None,
            meta=info,
            target_image_idxs=target_image_idxs,
        )

        if generation_kwargs is not None:
            for k, v in generation_kwargs.items():
                _data[k] = v

        data.append(_data)

    return data


def update_texts(
    inputs,
    text_ids,
    special_token_dict: dict = dict(
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=31999,
        soi_token_id=32000,
        image_token_id=32001,
    ),
    num_img_token=64,
    pad_image_tensor=None,
    force_gen_image_next=False,
    force_replace_gen_text=False,
    tokenizer=None,
    cur_iter=-1,
):
    gen_image_next = False
    stopped = False

    if force_replace_gen_text and "multiround_context" in inputs["meta"]:
        assert tokenizer is not None
        assert cur_iter >= 0
        text = inputs["meta"]["multiround_context"][cur_iter]
        if len(text) > 0:
            text_tensor = tokenizer(
                text,
                max_length=2048,
                truncation=False,
                padding=False,
                return_tensors="pt",
                return_attention_mask=True,
            )
            text_ids = text_tensor["input_ids"].to(device=text_ids.device)

    assert text_ids.shape[0] == 1
    text_ids = text_ids[0][1:]  # remove <bos>

    if text_ids[-1] == special_token_dict["eos_token_id"]:
        text_ids = text_ids[:-1]
        stopped = True
    if force_gen_image_next and text_ids[-1] != special_token_dict["soi_token_id"]:
        soi_id = torch.tensor([special_token_dict["soi_token_id"]]).type_as(text_ids)
        text_ids = torch.cat((text_ids, soi_id), dim=-1)

    if text_ids[-1] == special_token_dict["soi_token_id"]:
        image_ids = [special_token_dict["image_token_id"]] * num_img_token
        image_ids = torch.tensor(image_ids).type_as(text_ids)
        text_ids = torch.cat((text_ids, image_ids), dim=-1)
        # image_tensors, target_image_idxs, num_image_per_seq
        pad_image_tensor = pad_image_tensor.to(device=inputs["image_tensors"].device)

        inputs["image_tensors"] = torch.cat(
            [inputs["image_tensors"], pad_image_tensor], dim=0
        )
        inputs["target_image_idxs"] = inputs["target_image_idxs"] + 1
        inputs["num_image_per_seq"] = inputs["num_image_per_seq"] + 1
        gen_image_next = True

    text_ids = text_ids.unsqueeze(0)
    new_attn_mask = torch.ones_like(text_ids)

    inputs["text_ids"] = torch.cat((inputs["text_ids"], text_ids), dim=-1)
    inputs["attention_mask"] = torch.cat(
        (inputs["attention_mask"], new_attn_mask), dim=-1
    )

    return gen_image_next, stopped


def update_image(inputs, images, transform=None):
    assert len(images) == 1
    pil_images = tensor_to_pil(images)
    image_tensor_pred = transform(pil_images[0])
    if isinstance(image_tensor_pred, np.ndarray):
        image_tensor_pred = torch.from_numpy(image_tensor_pred)

    # update: image_tensors
    inputs["image_tensors"][-1, ...] = image_tensor_pred


def inference_all(model, config, annt_path, output_dir):
    # prepare data
    tokenizer = init_tokenizer(config.tokenizer_path)
    transform = create_transform(**config.transform)

    data = load_annt_data(
        transform=transform,
        tokenizer=tokenizer,
        num_img_token=config.num_img_token,
        generation_kwargs=config.generation_kwargs,
        annt_path=annt_path,
    )
    H = transform.resolution
    pad_image_tensor = torch.ones((1, 3, H, H)) * 0.5

    eval_results = []
    suffix = datetime.now().strftime("%Y%m%d%H%M")
    image_save_dir = os.path.join(output_dir, f"gen_img_{suffix}")
    os.makedirs(image_save_dir, exist_ok=True)

    print("Inference Start")
    for sample_idx, inputs in enumerate(data):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device="cuda")
                inputs[k] = v
        meta = inputs.get("meta", {})

        generate_mode = meta.get("generate_mode", config.generate_mode)
        if config.auto_end:
            max_num_iter = config.num_iter
        else:
            max_num_iter = meta.get("num_iter", config.num_iter)
        is_stopped = False

        meta["generate_results"] = []

        cur_iter = 0
        while cur_iter < max_num_iter:
            with torch.no_grad():
                outputs = model.generate(mode=generate_mode, **inputs)

            if generate_mode == "generate_texts":
                generate_texts = tokenizer.batch_decode(
                    outputs["text_ids"], skip_special_tokens=True
                )
                meta["generate_results"].append(generate_texts[0])
                gen_image_next, is_stopped = update_texts(
                    inputs,
                    outputs["text_ids"],
                    num_img_token=config.num_img_token,
                    pad_image_tensor=pad_image_tensor,
                    force_gen_image_next=config.force_gen_image_next,
                    force_replace_gen_text=config.force_replace_gen_text,
                    tokenizer=tokenizer,
                    cur_iter=cur_iter,
                )
                if gen_image_next:
                    generate_mode = "generate_images"
            elif generate_mode == "generate_images":
                for i in range(len(outputs["image"])):
                    image_fn = f"{sample_idx}_{cur_iter}_{i}.png"
                    save_image(
                        outputs["image"][i],
                        os.path.join(image_save_dir, image_fn),
                    )
                    meta["generate_results"].append(image_fn)
                update_image(inputs, outputs["image"][:1], transform=transform)
                generate_mode = "generate_texts"

            cur_iter += 1

            if config.auto_end and is_stopped:
                break

        eval_results.append(meta)

    with open(osp.join(output_dir, f"eval_results_{suffix}.json"), "w") as wf:
        json.dump(eval_results, wf, indent=4)

    print("All finished")


def main():
    parser = ArgumentParser(TrainingArguments)
    init_distributed_mode()
    args = parser.parse_args_with_config_file_into_dataclasses()
    train_args, config = args
    print(train_args)
    print(config)

    print("Model Init Start")
    model = MMInterleaved(**config.model)

    if getattr(config, "load_from", None):
        load_model_weights(model, config.load_from)
    model = model.to(device="cuda")
    model.eval()

    inference_all(model=model, config=config.inference, annt_path=config.annt_path, output_dir=train_args.output_dir)


if __name__ == "__main__":
    main()
