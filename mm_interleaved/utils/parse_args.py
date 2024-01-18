import os
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, fields

from mmcv import Config
import transformers
from transformers.hf_argparser import HfArgumentParser, DataClass

from .misc import is_main_process


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    config_file: Optional[str] = field(default="./configs/debug.yaml")
    resume: Optional[bool] = field(default=True)

    output_dir: Optional[str] = field(default="./OUTPUT/debug")
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )

    lr_for_random_params: Optional[float] = field(default=1e-3)
    random_params: Optional[str] = field(default=None)
    lr_for_random_params_list: Optional[List[str]]= field(default_factory=lambda: None)
    wd_for_random_params_list: Optional[List[str]]= field(default_factory=lambda: None)
    random_params_list: Optional[List[str]] = field(default_factory=lambda: None)

    generate_mode: Optional[str] = field(default="generate_texts")
    use_1st_sentence_only: Optional[bool] = field(default=False)


class ArgumentParser(HfArgumentParser):
    def parse_args_with_config_file_into_dataclasses(
        self,
        args=None,
        return_remaining_strings=False,
    ) -> Tuple[DataClass, ...]:
        """
        1. parse system arguments
        2. load yaml config file
        3. merge arguments from 2. into 1.,
        note that if there exists same arguments in both 2. and 1.,
        then the arguments in 1. will be overwritten by that in 2.
        4. split into different dataclasses
        """
        namespace, remaining_args = self.parse_known_args(args=args)
        config_file = getattr(namespace, "config_file", "./configs/debug.yaml")
        config_args = Config.fromfile(config_file)
        namespace.__dict__.update(config_args)
        if is_main_process():
            Config.dump(Config(namespace.__dict__), file=os.path.join(namespace.output_dir, "config.yaml"))

        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in fields(dtype) if f.init}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                delattr(namespace, k)
            obj = dtype(**inputs)
            outputs.append(obj)
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        if return_remaining_strings:
            return (*outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {remaining_args}")

            return (*outputs,)
