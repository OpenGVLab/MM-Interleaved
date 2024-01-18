from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Mapping,
)

import os
import sys
import time
import math
import glob
import warnings
import random
import shutil
import numpy as np
from packaging import version
from functools import partial
import webdataset as wds
import json
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

import diffusers
import datasets
import transformers
from transformers import CLIPProcessor, CLIPModel
from transformers.integrations import hp_params
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import (
    ShardedDDPOption,
    EvalPrediction,
    EvalLoopOutput,
    HPSearchBackend,
    TrainOutput,
    IntervalStrategy,
    PREFIX_CHECKPOINT_DIR,
    seed_worker,
    speed_metrics,
    has_length,
    denumpify_detensorize,
)
from transformers.trainer_pt_utils import (
    find_batch_size,
    get_model_param_count,
    get_parameter_names,
    reissue_pt_warnings,
    nested_concat,
    nested_numpify,
    IterableDatasetShard,
    LengthGroupedSampler,
)
from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.utils import (
    logging,
    is_sagemaker_mp_enabled,
    is_accelerate_available,
    is_apex_available,
    is_torch_tpu_available,
    is_datasets_available,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.trainer import (
    TRAINER_STATE_NAME,
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    SCALER_NAME,
)
from fairscale.optim import OSS

from ..utils import collect_caption_result, collect_vqa_result
from ..utils.fid_score import calculate_fid_given_paths
from ..utils.segm_eval import calculate_segm, calculate_miou_given_paths
from ..utils.clip_sim_score import (
    calculate_clip_sim_i2i,
    clip_rerank_generated_images,
    tensor_to_pil,
)
from ..utils.coco_cap_score import coco_caption_eval
from ..utils.visdial_metrics import scores_to_ranks, NDCG
from ..utils.vqa_score import vqa_eval, vizwiz_vqa_eval
from ..utils.grounding_score import grounding_eval
from ..custom_datasets.mmc4_wds import WdsDataset
from ..custom_datasets.mix_dataset import RandomMixWdsDataset


logger = logging.get_logger(__name__)

if is_apex_available():
    from apex import amp

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        GradientAccumulationPlugin,
    )

    if version.parse(accelerate_version) > version.parse("0.20.3"):
        from accelerate.utils import (
            load_fsdp_model,
            load_fsdp_optimizer,
            save_fsdp_model,
            save_fsdp_optimizer,
        )

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import (
        smp_forward_backward,
        smp_forward_only,
        smp_gather,
        smp_nested_concat,
    )
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


class LMMTrainer(transformers.Trainer):
    def __init__(self, *args, config=None, eval_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.eval_collator = eval_collator or self.data_collator
        self.generate_mode = self.args.generate_mode

    def _prepare_input(
        self, data: Union[torch.Tensor, Any]
    ) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (
                torch.is_floating_point(data) or torch.is_complex(data)
            ):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(
                    {
                        "dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()
                    }
                )

            return data.to(**kwargs)
        return data

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # HACK for pre-training with WdsDataset
        if isinstance(self.train_dataset, wds.DataPipeline) or isinstance(
            self.train_dataset, RandomMixWdsDataset
        ):
            print("Using WebLoader for training")
            dataloader = wds.WebLoader(
                self.train_dataset,
                batch_size=None,
                shuffle=False,
                num_workers=self.args.dataloader_num_workers,
                persistent_workers=(self.args.dataloader_num_workers > 0),
            )
            global_batch_size = (
                self.args.per_device_train_batch_size * self.accelerator.num_processes
            )
            dataloader.with_length(len(self.train_dataset) // global_batch_size)
            if has_length(dataloader):
                print(f"Length of dataloader: {len(dataloader)}")
            return dataloader

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": True,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """

        random_param_names = None
        if self.args.random_params_list is not None:
            random_param_names = self.args.random_params_list
        elif self.args.random_params is not None:
            random_param_names = [self.args.random_params]

        if random_param_names is None:
            return super().create_optimizer()

        lr_for_random_params = None
        wd_for_random_params = None
        if self.args.lr_for_random_params_list is not None:
            lr_for_random_params = self.args.lr_for_random_params_list
        elif self.args.lr_for_random_params is not None:
            lr_for_random_params = [self.args.lr_for_random_params] * len(
                random_param_names
            )

        if self.args.wd_for_random_params_list is not None:
            wd_for_random_params = self.args.wd_for_random_params_list
        else:
            wd_for_random_params = [None] * len(lr_for_random_params)

        assert len(lr_for_random_params) == len(random_param_names)
        print(
            f"use seperate lr {lr_for_random_params} for random init params {random_param_names}"
        )
        print(
            f"use seperate wd {wd_for_random_params} for random init params {random_param_names}"
        )

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            optimizer_grouped_parameters = [
                dict(params=[], weight_decay=0.0),
                dict(params=[], weight_decay=self.args.weight_decay),
            ]
            for i in range(len(random_param_names)):
                optimizer_grouped_parameters.extend(
                    [
                        dict(params=[], lr=lr_for_random_params[i], weight_decay=0.0),
                        dict(
                            params=[],
                            lr=lr_for_random_params[i],
                            weight_decay=self.args.weight_decay
                            if wd_for_random_params[i] is None
                            else wd_for_random_params[i],
                        ),
                    ]
                )

            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            random_params = {
                random_param_name: [] for random_param_name in random_param_names
            }
            for n, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue

                is_random = False
                for idx, random_param_name in enumerate(random_param_names):
                    if random_param_name in n:
                        random_params[random_param_name].append(n)
                        group_idx = (idx + 1) * 2 + int(n in decay_parameters)
                        optimizer_grouped_parameters[group_idx]["params"].append(p)
                        is_random = True
                        break
                if not is_random:
                    group_idx = int(n in decay_parameters)
                    optimizer_grouped_parameters[group_idx]["params"].append(p)
            print("detected random_params:", random_params)

            for idx, group in enumerate(optimizer_grouped_parameters):
                lr = group.get("lr", "-")
                wd = group["weight_decay"]
                total = [p.numel() for p in group["params"]]
                print(f"Group {idx}: lr {lr}, wd {wd}, params {total}")

            optimizer_cls, optimizer_kwargs = LMMTrainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(
                    optimizer_grouped_parameters, **optimizer_kwargs
                )
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum(
                                {
                                    p.data_ptr(): p.numel() for p in module.parameters()
                                }.values()
                            )
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(
                                module, "weight", {"optim_bits": 32}
                            )
                            logger.debug(
                                f"bitsandbytes: will optimize {module} in fp32"
                            )
                    logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        if self.is_deepspeed_enabled:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.model_wrapped.save_checkpoint(output_dir)

            # HACK for saving lr_scheduler as well
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(
                    self.lr_scheduler.state_dict(),
                    os.path.join(output_dir, SCHEDULER_NAME),
                )
            reissue_pt_warnings(caught_warnings)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        if self.fsdp or self.is_fsdp_enabled:
            if self.is_fsdp_enabled:
                save_fsdp_optimizer(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    self.optimizer,
                    self.model,
                    output_dir,
                )
            else:
                # FSDP has a different interface for saving optimizer states.
                # Needs to be called on all ranks to gather all states.
                # full_optim_state_dict will be deprecated after Pytorch 2.2!
                full_osd = self.model.__class__.full_optim_state_dict(
                    self.model, self.optimizer
                )

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(
                self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME)
            )
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(
                    self.lr_scheduler.state_dict(),
                    os.path.join(output_dir, SCHEDULER_NAME),
                )
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
            if self.args.should_save:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(
                        self.lr_scheduler.state_dict(),
                        os.path.join(output_dir, SCHEDULER_NAME),
                    )
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling:
                    torch.save(
                        self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME)
                    )
        elif self.args.should_save and not self.is_deepspeed_enabled:
            # deepspeed.save_checkpoint above saves model/optim/sched
            if self.fsdp and not self.is_fsdp_enabled:
                torch.save(full_osd, os.path.join(output_dir, OPTIMIZER_NAME))
            else:
                torch.save(
                    self.optimizer.state_dict(),
                    os.path.join(output_dir, OPTIMIZER_NAME),
                )

            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(
                    self.lr_scheduler.state_dict(),
                    os.path.join(output_dir, SCHEDULER_NAME),
                )
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(
                    self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME)
                )

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(
                rng_states,
                os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"),
            )

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        if self.is_deepspeed_enabled:
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init

            # HACK for loading lr_scheduler as well
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(
                    torch.load(os.path.join(checkpoint, SCHEDULER_NAME))
                )
            reissue_pt_warnings(caught_warnings)

            return

        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, OPTIMIZER_NAME) + "_*")
            if is_sagemaker_mp_enabled()
            else os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME))
        )
        if checkpoint_file_exists and os.path.isfile(
            os.path.join(checkpoint, SCHEDULER_NAME)
        ):
            # Load in optimizer and scheduler states
            if is_torch_tpu_available():
                # On TPU we have to take some extra precautions to properly load the states on the right device.
                optimizer_state = torch.load(
                    os.path.join(checkpoint, OPTIMIZER_NAME), map_location="cpu"
                )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    lr_scheduler_state = torch.load(
                        os.path.join(checkpoint, SCHEDULER_NAME), map_location="cpu"
                    )
                reissue_pt_warnings(caught_warnings)

                xm.send_cpu_data_to_device(optimizer_state, self.args.device)
                xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)

                self.optimizer.load_state_dict(optimizer_state)
                self.lr_scheduler.load_state_dict(lr_scheduler_state)
            else:
                if is_sagemaker_mp_enabled():
                    if os.path.isfile(os.path.join(checkpoint, "user_content.pt")):
                        # Optimizer checkpoint was saved with smp >= 1.10
                        def opt_load_hook(mod, opt):
                            opt.load_state_dict(
                                smp.load(
                                    os.path.join(checkpoint, OPTIMIZER_NAME),
                                    partial=True,
                                )
                            )

                    else:
                        # Optimizer checkpoint was saved with smp < 1.10
                        def opt_load_hook(mod, opt):
                            if IS_SAGEMAKER_MP_POST_1_10:
                                opt.load_state_dict(
                                    smp.load(
                                        os.path.join(checkpoint, OPTIMIZER_NAME),
                                        partial=True,
                                        back_compat=True,
                                    )
                                )
                            else:
                                opt.load_state_dict(
                                    smp.load(
                                        os.path.join(checkpoint, OPTIMIZER_NAME),
                                        partial=True,
                                    )
                                )

                    self.model_wrapped.register_post_step_hook(opt_load_hook)
                else:
                    # We use the CPU when training on one GPU to avoid OOM for GPU RAM when training big models.
                    # In distributed training however, we load directly on each GPU and risk the GPU OOM as it's more
                    # likely to get OOM on CPU (since we load num_gpu times the optimizer state
                    map_location = (
                        self.args.device if self.args.world_size > 1 else "cpu"
                    )
                    if self.fsdp or self.is_fsdp_enabled:
                        if self.is_fsdp_enabled:
                            load_fsdp_optimizer(
                                self.accelerator.state.fsdp_plugin,
                                self.accelerator,
                                self.optimizer,
                                self.model,
                                checkpoint,
                            )
                        else:
                            full_osd = None
                            # In FSDP, we need to load the full optimizer state dict on rank 0 and then shard it
                            if self.args.process_index == 0:
                                full_osd = torch.load(
                                    os.path.join(checkpoint, OPTIMIZER_NAME)
                                )
                            # call scatter_full_optim_state_dict on all ranks
                            sharded_osd = (
                                self.model.__class__.scatter_full_optim_state_dict(
                                    full_osd, self.model
                                )
                            )
                            self.optimizer.load_state_dict(sharded_osd)
                    else:
                        self.optimizer.load_state_dict(
                            torch.load(
                                os.path.join(checkpoint, OPTIMIZER_NAME),
                                map_location=map_location,
                            )
                        )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.lr_scheduler.load_state_dict(
                        torch.load(os.path.join(checkpoint, SCHEDULER_NAME))
                    )
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling and os.path.isfile(
                    os.path.join(checkpoint, SCALER_NAME)
                ):
                    self.scaler.load_state_dict(
                        torch.load(os.path.join(checkpoint, SCALER_NAME))
                    )

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(
                model, inputs, self.args.gradient_accumulation_steps
            )
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        #     labels = None

        outputs = model(**inputs)

        assert isinstance(outputs, dict)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs[self.args.past_index]

        # if labels is not None:
        #     if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, only the following keys: "
        #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss: torch.Tensor = outputs.pop('loss') if isinstance(outputs, dict) else outputs[0]

        loss: torch.Tensor = outputs.pop("loss")

        return (loss, outputs) if return_outputs else loss

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(
            f"Currently training with a batch size of: {self._train_batch_size}"
        )
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = (
            self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        )

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = (
                len_dataloader // args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = (
                    self.num_examples(train_dataloader) * args.num_train_epochs
                )
        elif (
            args.max_steps > 0
        ):  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps and args.logging_steps < 1:
            args.logging_steps = math.ceil(max_steps * args.logging_steps)
        if args.eval_steps and args.eval_steps < 1:
            args.eval_steps = math.ceil(max_steps * args.eval_steps)
        if args.save_steps and args.save_steps < 1:
            args.save_steps = math.ceil(max_steps * args.save_steps)

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps
            )

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(
                        self.model, self.optimizer
                    )
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # deepspeed ckpt loading
        if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
            print(
                f"deepspeed resume checkpoint from {resume_from_checkpoint}, {type(self.model_wrapped)=}"
            )
            deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            print(f"{self.model_wrapped.optimizer=} {self.model_wrapped.lr_scheduler=}")

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}"
        )
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(
                f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}"
            )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = (
                trial.assignments
                if self.hp_search_backend == HPSearchBackend.SIGOPT
                else trial
            )
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                for _ in train_dataloader:
                    break

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            # HACK for using Wdsdataset
            if isinstance(train_dataloader, wds.WebLoader):
                dataset = train_dataloader.pipeline[0].dataset
                if isinstance(dataset, WdsDataset):
                    dataset.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            if (
                epoch == epochs_trained
                and resume_from_checkpoint is not None
                and steps_trained_in_current_epoch == 0
            ):
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                steps_skipped = steps_trained_in_current_epoch
                if not isinstance(epoch_iterator, wds.WebLoader):  # HACK by TCY
                    epoch_iterator = skip_first_batches(
                        epoch_iterator, steps_trained_in_current_epoch
                    )
                    steps_trained_in_current_epoch = 0
                    rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    # HACK for skipping data for WdsDataset
                    if (
                        self.args.logging_strategy == IntervalStrategy.STEPS
                        and steps_trained_in_current_epoch % self.args.logging_steps
                        == 0
                    ):
                        print(
                            f"Steps_trained to be skipped: {steps_trained_in_current_epoch}"
                        )
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                else:
                    # HACK here
                    # tr_loss += tr_loss_step
                    tr_loss = tr_loss + tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc or (
                        version.parse(accelerate_version) <= version.parse("0.20.3")
                    ):
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            # if is_torch_tpu_available():
                            #     gradients = xm._fetch_gradients(self.optimizer)
                            #     xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        # elif self.use_apex:
                        #     # Revert to normal clipping otherwise, handling Apex or full precision
                        #     nn.utils.clip_grad_norm_(
                        #         amp.master_params(self.optimizer),
                        #         args.max_grad_norm,
                        #     )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            # tpu-comment: accelerate wrapped optimizers call xm.optimizer_step
                            self.optimizer.step()
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                        optimizer_was_run = (
                            not self.accelerator.optimizer_step_was_skipped
                        )

                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(
                            self.lr_scheduler,
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                        ):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = (
                        epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    )
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )

                    self._maybe_log_save_evaluate(
                        tr_loss, model, trial, epoch, ignore_keys_for_eval
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break


            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss, model, trial, epoch, ignore_keys_for_eval
            )

            # if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            #     if is_torch_tpu_available():
            #         # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            #         xm.master_print(met.metrics_report())
            #     else:
            #         logger.warning(
            #             "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
            #             "configured. Check your training configuration if this is unexpected."
            #         )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            # if is_torch_tpu_available():
            #     xm.rendezvous("load_best_model_at_end")
            # elif args.parallel_mode == ParallelMode.DISTRIBUTED:
            #     dist.barrier()
            # elif is_sagemaker_mp_enabled():
            #     smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir
        )

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if (
            self.args.should_save
            and self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                    )
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = getattr(eval_dataset, "collator", None) or self.eval_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="evaluation"
            )

        dataloader_params = {
            "batch_size": getattr(
                eval_dataset, "batch_size", self.args.eval_batch_size
            ),
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def _inner_generation_loop(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        generate_mode: str = "",
        metric_key_prefix="",
        num_samples: int = 0,
        annt_file="",
        use_1st_sentence_only=False,
        rerank_by_clip=False,
        clip_model_name="./assets/openai/clip-vit-large-patch14",
        **kwargs,
    ) -> EvalLoopOutput:
        assert num_samples > 0

        args = self.args

        generate_mode = generate_mode or self.generate_mode
        assert generate_mode in [
            "generate_texts",
            "generate_images",
            "generate_vqa",
            "generate_segm",
            "generate_grounding",
        ]

        batch_size = self.args.eval_batch_size

        save_path = os.path.join(
            args.output_dir,
            f"{metric_key_prefix}_{generate_mode}",
            f"ckpt-{self.state.global_step}-{self.state.epoch}",
        )
        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
        self.accelerator.wait_for_everyone()

        print(f"Eval Loop Start mode: {generate_mode} num_samples {num_samples}")

        if rerank_by_clip:
            vis_img = True
            clip_model = CLIPModel.from_pretrained(clip_model_name)
            clip_model.to("cuda")
            clip_model.eval()
            clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        caption_result = []
        observed_num_examples = 0
        for step, inputs in enumerate(dataloader):
            print(f"Eval iter: {step} / {len(dataloader)}")
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # currently we only consider generate either image or text in evaluation
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    outputs = model.generate(mode=generate_mode, **inputs)

            if generate_mode in ("generate_texts", "generate_vqa"):
                generate_texts = self.tokenizer.batch_decode(
                    outputs["text_ids"], skip_special_tokens=True
                )
                data_index = [d[0] for d in inputs["meta"]]
                for text, sample_id in zip(generate_texts, data_index):
                    caption_result.append(
                        {"image_id": sample_id, "caption": text.strip()}
                    )
            elif generate_mode == "generate_images":
                data_index = [d[0] for d in inputs["meta"]]
                captions = [d[1] for d in inputs["meta"]]
                if rerank_by_clip:
                    if vis_img and self.accelerator.is_main_process:
                        print(captions)
                        print(f"{outputs['image'].shape=}")
                        save_image(
                            outputs["image"],
                            os.path.join(args.output_dir, "clip_rerank_before.png"),
                        )

                    outputs["image"] = clip_rerank_generated_images(
                        outputs["image"],
                        captions,
                        clip_model,
                        clip_processor,
                        device="cuda",
                    )

                    if vis_img and self.accelerator.is_main_process:
                        print(f"{len(outputs['image'])=}")
                        for i, image in enumerate(outputs["image"]):
                            image.save(
                                os.path.join(
                                    args.output_dir, f"clip_rerank_after_{i}.png"
                                )
                            )
                        vis_img = False

                for sample_id, (image, caption, data_id) in enumerate(
                    zip(outputs["image"], captions, data_index)
                ):
                    idx = (
                        (step * batch_size + sample_id) * self.accelerator.num_processes
                        + self.accelerator.process_index
                    )
                    if idx < num_samples:
                        caption_result.append(
                            {"image_id": idx, "caption": caption, "data_id": data_id}
                        )
                        if isinstance(image, Image.Image):
                            image.save(os.path.join(save_path, f"{idx:05d}.png"))
                        else:
                            save_image(image, os.path.join(save_path, f"{idx:05d}.png"))
            elif generate_mode == "generate_grounding":
                generate_texts = self.tokenizer.batch_decode(
                    outputs["text_ids"], skip_special_tokens=True
                )
                data_index = [d[0] for d in inputs["meta"]]
                for text, meta in zip(generate_texts, inputs["meta"]):
                    h, w, gt_box = meta[-3:]
                    caption_result.append(
                        {
                            "gt_box": gt_box,
                            "pred_box": text.strip(),
                            "height": h,
                            "width": w,
                        }
                    )
            elif generate_mode == "generate_segm":
                data_index = [d[0] for d in inputs["meta"]]
                captions = [d[1] for d in inputs["meta"]]
                for sample_id, (image, caption, data_id) in enumerate(
                    zip(outputs["image"], captions, data_index)
                ):
                    idx = (
                        (step * batch_size + sample_id) * self.accelerator.num_processes
                        + self.accelerator.process_index
                    )
                    if data_id < num_samples:
                        caption_result.append(
                            {"image_id": idx, "caption": caption, "data_id": data_id}
                        )

                        gt_img = Image.open(dataloader.dataset.gt_id_to_path(data_id))

                        image = (
                            image.clone()
                            .mul(255)
                            .add_(0.5)
                            .clamp_(0, 255)
                            .permute(1, 2, 0)
                            .to("cpu", torch.uint8)
                            .numpy()
                        )
                        image = Image.fromarray(image).resize(gt_img.size)

                        segm_image = calculate_segm(image, gt_img)
                        segm_image = segm_image.to("cpu", torch.uint8).numpy()
                        segm_image = Image.fromarray(segm_image)
                        segm_image.putpalette(dataloader.dataset.palette)

                        image.save(os.path.join(save_path, f"{data_id:06d}.png"))
                        segm_image.save(
                            os.path.join(save_path, f"{data_id:06d}_segm.png")
                        )

        self.accelerator.wait_for_everyone()
        print(f"Eval Loop Finished")

        metrics = {}

        if generate_mode == "generate_texts":
            metric_keys = ["Bleu_4", "CIDEr"]
            caption_result_file = collect_caption_result(
                caption_result,
                save_path,
                "val_caption_pred",
                remove_duplicate="image_id",
            )
            if self.accelerator.is_main_process:
                _metrics = coco_caption_eval(
                    annt_file,
                    caption_result_file,
                    use_1st_sentence_only=use_1st_sentence_only,
                )
                for metric, score in _metrics.items():
                    print(f"{metric}: {score:.3f}")
                    if metric in metric_keys:
                        metrics[metric] = score
            else:
                metrics["Bleu_4"] = None
        elif generate_mode == "generate_images":
            metric_keys = ["FID"]
            # gather text descriptions for generated image
            caption_result_file = collect_caption_result(
                caption_result, save_path, "val_caption_gt", remove_duplicate="image_id"
            )
            fid = None
            if self.accelerator.is_main_process:
                image_paths = [
                    os.path.join(save_path, f"{idx:05d}.png")
                    for idx in range(num_samples)
                ]
                with open(caption_result_file, "r") as rf:
                    caption_result = json.load(rf)
                image_paths_gt = [
                    dataloader.dataset.image_id_to_path(caption_result[idx]["data_id"])
                    for idx in range(num_samples)
                ]
                fid = calculate_fid_given_paths((image_paths_gt, image_paths))
                print(f"generate FID: {fid}")
            metrics["FID"] = fid
        elif generate_mode == "generate_segm":
            metric_keys = ["mIoU"]
            caption_result_file = collect_caption_result(
                caption_result, save_path, "val_caption_gt", remove_duplicate="image_id"
            )

            if self.accelerator.is_main_process:
                with open(caption_result_file, "r") as rf:
                    caption_result = json.load(rf)
                image_paths = [
                    os.path.join(
                        save_path, f"{caption_result[idx]['data_id']:06d}_segm.png"
                    )
                    for idx in range(num_samples)
                ]
                image_paths_gt = [
                    dataloader.dataset.gt_id_to_path(caption_result[idx]["data_id"])
                    for idx in range(num_samples)
                ]
                iou = calculate_miou_given_paths((image_paths_gt, image_paths))

                metrics["IoU"] = iou

        elif generate_mode == "generate_vqa":
            metric_keys = ["overall_accuracy"]
            caption_result_file = collect_vqa_result(
                caption_result,
                save_path,
                f"val_{dataloader.dataset.__class__.__name__}_pred",
                is_vizwiz=not hasattr(dataloader.dataset, "question_file"),
            )
            if self.accelerator.is_main_process:
                eval_func = (
                    partial(vqa_eval, question_file=dataloader.dataset.question_file)
                    if hasattr(dataloader.dataset, "question_file")
                    else vizwiz_vqa_eval
                )
                _metrics = eval_func(
                    annotation_file=annt_file,
                    results_file=caption_result_file,
                    use_extract_answer=True,
                )
                for metric, score in _metrics.items():
                    print(f"{metric}: {score:.3f}")
                    if metric in metric_keys:
                        metrics[metric] = score
        elif generate_mode == "generate_grounding":
            metric_keys = ["accuracy"]
            grounding_result_file = collect_caption_result(
                caption_result,
                save_path,
                "grounding_pred",
            )
            if self.accelerator.is_main_process:
                _metrics = grounding_eval(grounding_result_file)
                for metric, score in _metrics.items():
                    print(f"{metric}: {score:.3f}")
                    if metric in metric_keys:
                        metrics[metric] = score
        self.accelerator.wait_for_everyone()

        if num_samples != observed_num_examples:
            print(
                f"[WARNING] The real evaluation sample {observed_num_examples} is not equal to expected num_samples {num_samples}"
                f", which may because of distributed evaluation."
            )

        return EvalLoopOutput(
            predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples
        )

    def _inner_generation_loop_v2(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        generate_mode: str = "",
        metric_key_prefix="",
        num_samples: int = 0,
        save_gt_image_online=False,
        **kwargs,
    ) -> EvalLoopOutput:
        assert num_samples > 0
        args = self.args
        generate_mode = generate_mode or self.generate_mode

        assert generate_mode in ["generate_images"]
        batch_size = self.args.eval_batch_size

        save_path = os.path.join(
            args.output_dir,
            f"{metric_key_prefix}_{generate_mode}",
            f"ckpt-{self.state.global_step}-{self.state.epoch}",
        )
        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
        if save_gt_image_online:
            save_path_gt = os.path.join(
                args.output_dir,
                f"{metric_key_prefix}_{generate_mode}_gt",
                f"ckpt-{self.state.global_step}-{self.state.epoch}",
            )
            if self.accelerator.is_main_process:
                os.makedirs(save_path_gt, exist_ok=True)

        target_image_idxs_per_sample = getattr(
            dataloader.dataset, "target_image_idxs", None
        )

        self.accelerator.wait_for_everyone()

        print(f"Eval Loop Start mode: {generate_mode} num_samples {num_samples}")

        generation_result = []
        observed_num_examples = 0
        for step, inputs in enumerate(dataloader):
            print(f"Eval iter: {step} / {len(dataloader)}")
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # currently we only consider generate either image or text in evaluation
            inputs = self._prepare_inputs(inputs)

            if target_image_idxs_per_sample is not None:
                assert save_gt_image_online
                for target_image_idx in target_image_idxs_per_sample:
                    num_image_per_seq = inputs["num_image_per_seq"]
                    target_image_idxs = (
                        torch.cat(
                            (
                                torch.zeros_like(num_image_per_seq[:1]),
                                torch.cumsum(num_image_per_seq, dim=0)[:-1],
                            ),
                            dim=0,
                        )
                        + target_image_idx
                    )
                    inputs["target_image_idxs"] = target_image_idxs
                    print(f"{target_image_idx=} {target_image_idxs=}")

                    with torch.no_grad():
                        with self.compute_loss_context_manager():
                            outputs = model.generate(mode=generate_mode, **inputs)

                    # auto-regressive
                    pil_images = tensor_to_pil(outputs["image"])
                    images_tensors_all = []
                    for image in pil_images:
                        image_tensor_pred = dataloader.dataset.transform(image)
                        if isinstance(image_tensor_pred, np.ndarray):
                            image_tensor_pred = torch.from_numpy(image_tensor_pred)
                        images_tensors_all.append(image_tensor_pred)
                    image_tensors_pred = torch.stack(images_tensors_all, dim=0)
                    image_tensors_pred = self._prepare_inputs(image_tensors_pred)
                    image_tensors = inputs["image_tensors"]
                    C, H, W = image_tensors[0].shape
                    target_image_idxs = target_image_idxs[:, None, None, None].expand(
                        -1, C, H, W
                    )
                    image_tensors = torch.scatter(
                        image_tensors,
                        dim=0,
                        index=target_image_idxs,
                        src=image_tensors_pred,
                    )
                    inputs["image_tensors"] = image_tensors

                    for sample_id, (image, meta) in enumerate(
                        zip(outputs["image"], inputs["meta"])
                    ):
                        image_id = (
                            (step * batch_size + sample_id)
                            * self.accelerator.num_processes
                            + self.accelerator.process_index
                        )
                        if image_id < num_samples:
                            image_path = os.path.join(
                                save_path, f"{image_id:05d}_{target_image_idx:02d}.png"
                            )
                            save_image(image, image_path)
                            image_gt_path = meta[1]
                            if save_gt_image_online:
                                image_gt: Image.Image = (
                                    dataloader.dataset.meta_to_image(
                                        meta,
                                        target_image_idx=target_image_idx,
                                    )
                                )
                                image_gt_path = os.path.join(
                                    save_path_gt,
                                    f"{image_id:05d}_{target_image_idx:02d}.png",
                                )
                                image_gt.save(image_gt_path)
                            _result = {
                                "sample_id": meta[0],
                                "frame_idxs": meta[1],
                                "image_id": image_id,
                                "image_path": image_path,
                                "image_gt_path": image_gt_path,
                            }
                            generation_result.append(_result)
            else:
                with torch.no_grad():
                    with self.compute_loss_context_manager():
                        outputs = model.generate(mode=generate_mode, **inputs)

                for sample_id, (image, meta) in enumerate(
                    zip(outputs["image"], inputs["meta"])
                ):
                    image_id = (
                        (step * batch_size + sample_id)
                        * self.accelerator.num_processes
                        + self.accelerator.process_index
                    )
                    if image_id < num_samples:
                        image_path = os.path.join(save_path, f"{image_id:05d}.png")
                        save_image(image, image_path)
                        image_gt_path = meta[1]
                        if save_gt_image_online:
                            image_gt: Image.Image = (
                                dataloader.dataset.meta_to_image(meta)
                            )
                            image_gt_path = os.path.join(
                                save_path_gt, f"{image_id:05d}.png"
                            )
                            image_gt.save(image_gt_path)
                        _result = {
                            "sample_id": meta[0],
                            "frame_idxs": meta[1],
                            "image_id": image_id,
                            "image_path": image_path,
                            "image_gt_path": image_gt_path,
                        }
                        generation_result.append(_result)

        self.accelerator.wait_for_everyone()
        print(f"Eval Loop Finished")

        metrics = {}

        # calc metrics
        generation_result_file = collect_caption_result(
            generation_result,
            save_path,
            "val_generation_result",
            remove_duplicate="image_path",
        )

        self.accelerator.wait_for_everyone()
        with open(generation_result_file, "r") as rf:
            generation_result = json.load(rf)

        clip_score = calculate_clip_sim_i2i(generation_result, device="cuda")
        fid = None
        if self.accelerator.is_main_process:
            gt_paths = [r["image_gt_path"] for r in generation_result]
            pred_paths = [r["image_path"] for r in generation_result]
            fid = calculate_fid_given_paths((gt_paths, pred_paths))
            print(f"generate FID: {fid}")
        metrics["FID"] = fid
        metrics["clip_score_i2i"] = clip_score

        self.accelerator.wait_for_everyone()

        if num_samples != observed_num_examples:
            print(
                f"[WARNING] The real evaluation sample {observed_num_examples} is not equal to expected num_samples {num_samples}"
                f", which may because of distributed evaluation."
            )

        return EvalLoopOutput(
            predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples
        )

    def _inner_ranking_loop(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        generate_mode: str = "",
        metric_key_prefix="",
        num_samples: int = 0,
        dense_annt = None,
        mini_bs = 128,
        **kwargs,
    ) -> EvalLoopOutput:
        assert num_samples > 0

        args = self.args

        generate_mode = generate_mode or self.generate_mode
        assert generate_mode in ["generate_scores"]

        batch_size = self.args.eval_batch_size
        save_path = os.path.join(
            args.output_dir,
            f"{metric_key_prefix}_{generate_mode}",
            f"ckpt-{self.state.global_step}-{self.state.epoch}",
        )
        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
        self.accelerator.wait_for_everyone()

        print(f"Eval Loop Start mode: {generate_mode} num_samples {num_samples}")

        scores_result = []
        observed_num_examples = 0
        for step, inputs in enumerate(dataloader):
            print(f"Eval iter: {step} / {len(dataloader)}")
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # currently we only consider generate either image or text in evaluation
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    outputs = model.generate(mode=generate_mode, **inputs)

            predicted_ranks = scores_to_ranks(outputs["scores"])
            for image_id, rank in zip(inputs["image_ids"], predicted_ranks):
                scores_result.append(
                    {"image_id": image_id.item(), "rank": rank.cpu().tolist()}
                )

        print(f"Eval Loop Finished")

        metrics = {}

        filename = "val_rank_pred"
        result_file = os.path.join(
            save_path, "%s_rank%d.json" % (filename, self.accelerator.process_index)
        )
        with open(result_file, "w") as wf:
            json.dump(scores_result, wf)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            result = []

            for rank in range(dist.get_world_size()):
                result_file = os.path.join(
                    save_path, "%s_rank%d.json" % (filename, rank)
                )
                with open(result_file, "r") as rf:
                    res = json.load(rf)
                result.extend(res)
                os.remove(result_file)
            result_file = os.path.join(save_path, f"{filename}_rank.json")
            with open(result_file, "w") as wf:
                json.dump(result, wf)

            ndcg = NDCG()
            total_iter = math.ceil(len(result) / mini_bs)
            for i in range(total_iter):
                _result = result[i * mini_bs : (i + 1) * mini_bs]
                gt_relevance = torch.tensor(
                    [dense_annt[d["image_id"]]["gt_relevance"] for d in _result]
                )
                ranks = torch.tensor([d["rank"] for d in _result])
                ndcg.observe(target_relevance=gt_relevance, predicted_ranks=ranks)
            metrics.update(ndcg.retrieve(reset=True))

        self.accelerator.wait_for_everyone()
        if num_samples != observed_num_examples:
            print(
                f"[WARNING] The real evaluation sample {observed_num_examples} is not equal to expected num_samples {num_samples}"
                f", which may because of distributed evaluation."
            )
        return EvalLoopOutput(
            predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples
        )

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        """
            Modify the original pipeline from here
        """

        num_samples = getattr(args, "num_samples", 0)
        if num_samples == 0:
            # Number of samples
            if has_length(eval_dataset):
                num_samples = len(eval_dataset)
            # The instance check is weird and does not actually check for the type, but whether the dataset has the right
            # methods. Therefore we need to make sure it also has the attribute.
            elif (
                isinstance(eval_dataset, IterableDatasetShard)
                and getattr(eval_dataset, "num_examples", 0) > 0
            ):
                num_samples = eval_dataset.num_examples
            else:
                if has_length(dataloader):
                    num_samples = self.num_examples(dataloader)
                else:
                    raise ValueError(
                        "Please specify total generated samples first for evaluation!"
                    )

        print(
            f"self.is_deepspeed_enabled: {self.is_deepspeed_enabled}; self.use_cuda_amp: "
            f"{self.use_cuda_amp}; self.half_precision_backend: {self.args.half_precision_backend}"
        )

        # decide what generate_mode to use based on eval_dataset
        generate_mode = getattr(eval_dataset, "collate_mode", self.generate_mode)
        print(f"{generate_mode} for {eval_dataset}")

        eval_metrics = {}

        dataset_cls_name = eval_dataset.__class__.__name__
        if dataset_cls_name in ["VISTDataset", "PororoDataset", "FlintStonesDataset"]:
            task_prefix = getattr(eval_dataset, "task_prefix", "")
            metric_key_prefix = metric_key_prefix + task_prefix
            eval_outputs = self._inner_generation_loop_v2(
                dataloader,
                model,
                generate_mode=generate_mode,
                metric_key_prefix=metric_key_prefix,
                num_samples=num_samples,
                save_gt_image_online=getattr(
                    eval_dataset,
                    "save_gt_image_online",
                    False,
                ),
            )
            eval_metrics = eval_outputs.metrics
        elif generate_mode in ("generate_texts", "generate_vqa", "generate_grounding"):
            eval_outputs = self._inner_generation_loop(
                dataloader,
                model,
                generate_mode=generate_mode,
                metric_key_prefix=metric_key_prefix,
                num_samples=num_samples,
                annt_file=eval_dataset.annt_file,
                use_1st_sentence_only=getattr(
                    eval_dataset,
                    "use_1st_sentence_only",
                    self.args.use_1st_sentence_only,
                ),
            )
            eval_metrics = eval_outputs.metrics
        elif generate_mode == "generate_images":
            eval_outputs = self._inner_generation_loop(
                dataloader,
                model,
                generate_mode=generate_mode,
                metric_key_prefix=metric_key_prefix,
                num_samples=num_samples,
                rerank_by_clip=getattr(eval_dataset, "rerank_by_clip", False),
            )
            eval_metrics = eval_outputs.metrics
        elif generate_mode == "generate_segm":
            eval_outputs = self._inner_generation_loop(
                dataloader,
                model,
                generate_mode=generate_mode,
                metric_key_prefix=metric_key_prefix,
                num_samples=num_samples,
                rerank_by_clip=getattr(eval_dataset, "rerank_by_clip", False),
            )
            eval_metrics = eval_outputs.metrics
        elif generate_mode == "generate_both":
            dataloader.collate_fn.set_mode("generate_texts")
            eval_outputs = self._inner_generation_loop(
                dataloader,
                model,
                generate_mode="generate_texts",
                metric_key_prefix=metric_key_prefix,
                num_samples=num_samples,
                annt_file=eval_dataset.annt_file,
                use_1st_sentence_only=getattr(
                    eval_dataset,
                    "use_1st_sentence_only",
                    self.args.use_1st_sentence_only,
                ),
            )
            eval_metrics.update(eval_outputs.metrics)
            dataloader.collate_fn.set_mode("generate_images")
            eval_outputs = self._inner_generation_loop(
                dataloader,
                model,
                generate_mode="generate_images",
                metric_key_prefix=metric_key_prefix,
                num_samples=num_samples,
            )
            eval_metrics.update(eval_outputs.metrics)
            dataloader.collate_fn.set_mode(generate_mode)
        elif generate_mode == "generate_scores":
            # for visdial evaluation
            eval_outputs = self._inner_ranking_loop(
                dataloader,
                model,
                generate_mode=generate_mode,
                metric_key_prefix=metric_key_prefix,
                num_samples=num_samples,
                dense_annt=eval_dataset.dense_annt,
            )
            eval_metrics = eval_outputs.metrics

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(eval_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                eval_metrics[f"{metric_key_prefix}_{key}"] = eval_metrics.pop(key)

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=eval_metrics,
            num_samples=num_samples,
        )

    def _maybe_log_save_evaluate(
        self, tr_loss, model, trial, epoch, ignore_keys_for_eval
    ):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # HACK for saving eval metrics to local json file
            if self.args.should_save:
                metrics_to_save = {**metrics, **{"step": self.state.global_step}}
                if self.state.epoch is not None:
                    metrics_to_save["epoch"] = round(self.state.epoch, 2)
                metrics_save_path = os.path.join(
                    self.args.output_dir, "eval_metrics.jsonl"
                )
                json_string = (
                    json.dumps(metrics_to_save, indent=2, sort_keys=True) + "\n"
                )
                with open(metrics_save_path, "a+", encoding="utf-8") as f:
                    f.write(json_string)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # HACK by commenting out speed metrics
        # output.metrics.update(
        #     speed_metrics(
        #         metric_key_prefix,
        #         start_time,
        #         num_samples=output.num_samples,
        #         num_steps=math.ceil(output.num_samples / total_batch_size),
        #     )
        # )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
