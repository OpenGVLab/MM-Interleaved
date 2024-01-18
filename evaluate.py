import json
import os

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
from mm_interleaved.custom_datasets.utils import build_dataset
from mm_interleaved.engine.lmm_trainer import LMMTrainer
from mm_interleaved.utils import ArgumentParser, TrainingArguments, init_distributed_mode, load_model_weights


def evaluate(trainer: LMMTrainer, config):
    print("Eval Start")
    if isinstance(trainer.eval_dataset, dict):
        eval_datasets = trainer.eval_dataset
    else:
        eval_datasets = {config.data.val.name: trainer.eval_dataset}

    metrics = {}
    for eval_dataset_name, eval_dataset in eval_datasets.items():
        dataset_metrics = trainer.evaluate(
            eval_dataset=eval_dataset,
            metric_key_prefix=f"eval_{eval_dataset_name}",
        )
        print(eval_dataset_name)
        print(dataset_metrics)
        print("-" * 100)
        metrics.update(dataset_metrics)
    print("=" * 100)

    if trainer.args.should_save:
        metrics_to_save = {
            **metrics,
            **{"step": trainer.state.global_step},
        }
        if trainer.state.epoch is not None:
            metrics_to_save["epoch"] = round(trainer.state.epoch, 2)
        metrics_save_path = os.path.join(trainer.args.output_dir, "eval_metrics.jsonl")
        json_string = json.dumps(metrics_to_save, indent=2, sort_keys=True) + "\n"
        with open(metrics_save_path, "a+", encoding="utf-8") as f:
            f.write(json_string)

    print("All Finished")


def main():
    parser = ArgumentParser(TrainingArguments)
    init_distributed_mode()
    args = parser.parse_args_with_config_file_into_dataclasses()
    train_args, config = args
    print(train_args)
    print(config)

    print("Data Loading Start")
    eval_dataset = build_dataset(config.data.val)
    print(eval_dataset)

    print("Model Init Start")
    model = MMInterleaved(**config.model)
    print(model)

    print("Trainer Init Start")
    if isinstance(eval_dataset, dict):
        tokenizer = list(eval_dataset.values())[0].tokenizer
    else:
        tokenizer = eval_dataset.tokenizer
    trainer = LMMTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        args=train_args,
        eval_dataset=eval_dataset,
    )

    if getattr(config, "load_from", None):
        load_model_weights(trainer.model, config.load_from)

    evaluate(trainer, config)


if __name__ == "__main__":
    main()
