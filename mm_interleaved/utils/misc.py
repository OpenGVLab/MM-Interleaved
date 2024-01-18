import math
import builtins
import datetime
import os
import time
from absl import logging
from collections import defaultdict, deque
import subprocess
import torch
import torch.distributed as dist


def load_model_weights(model, ckpt_path, image_upscale=1.0):
    print("loading:", ckpt_path)
    if os.path.isdir(ckpt_path):
        pretrained_weights = defaultdict()
        ckpt_files = os.listdir(ckpt_path)
        for ckpt_fn in ckpt_files:
            if ckpt_fn.endswith(".bin"):
                weights = torch.load(os.path.join(ckpt_path, ckpt_fn), "cpu")
                pretrained_weights.update(weights)
    else:
        pretrained_weights = torch.load(ckpt_path, "cpu")
    pos_embed_key = (
        "visual_tokenizer.encoder.vision_model.embeddings.position_embedding.weight"
    )
    is_vit_l = pos_embed_key in pretrained_weights.keys()
    if not is_vit_l:
        pos_embed_key = "visual_tokenizer.encoder.pos_embed"
    pos_embed_pretrained = pretrained_weights[pos_embed_key].float()
    if is_vit_l:
        pos_embed_pretrained = pos_embed_pretrained.unsqueeze(0)
    old_size = int(math.sqrt(pos_embed_pretrained.size(1) - 1))
    new_size = int(old_size * image_upscale)
    if old_size != new_size:
        cls = pos_embed_pretrained[:, :1, :]
        pos_embed = (
            pos_embed_pretrained[:, 1:, :]
            .reshape(1, old_size, old_size, -1)
            .permute(0, 3, 1, 2)
        )
        pos_embed = (
            torch.nn.functional.interpolate(
                pos_embed,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            .reshape(1, -1, new_size * new_size)
            .permute(0, 2, 1)
        )
        pos_embed_pretrained = torch.cat([cls, pos_embed], dim=1)
        if is_vit_l:
            pos_embed_pretrained = pos_embed_pretrained.squeeze(0)
        pretrained_weights[pos_embed_key] = pos_embed_pretrained
        position_ids_k = []
        for k in pretrained_weights.keys():
            if "position_ids" in k:
                position_ids_k.append(k)
        for k in position_ids_k:
            pretrained_weights.pop(k)
    message = model.load_state_dict(pretrained_weights, strict=False)
    print(message)


def set_logger(log_level="info", fname=None):
    import logging as _logging

    handler = logging.get_absl_handler()
    formatter = _logging.Formatter("%(asctime)s - %(filename)s - %(message)s")
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        # force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def barrier():
    if not is_dist_avail_and_initialized():
        return
    dist.barrier()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(use_dynamic_port: bool = False):
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = rank % torch.cuda.device_count()

        world_size = int(os.environ["SLURM_NTASKS"])
        try:
            local_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
        except:
            local_size = int(os.environ.get("LOCAL_SIZE", 1))

        if "MASTER_PORT" not in os.environ:
            port = 22110
            if use_dynamic_port:
                for i in range(22110, 65535):
                    cmd = f"netstat -aon|grep {i}"
                    with os.popen(cmd, "r") as file:
                        if file.read() == "":
                            port = i
                            break

            print(f"MASTER_PORT = {port}")
            os.environ["MASTER_PORT"] = str(port)

            time.sleep(3)

        node_list = os.environ["SLURM_STEP_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr

        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(local_size)
        os.environ["WORLD_SIZE"] = str(world_size)

    else:
        rank = int(os.environ["RANK"])

    setup_for_distributed(rank == 0)

    print(
        f"Rank {os.environ['RANK']} | Local Rank {os.environ['LOCAL_RANK']} | "
        f"World Size {os.environ['WORLD_SIZE']} | Local World Size {os.environ['LOCAL_WORLD_SIZE']} |",
        force=True,
    )
