import abc
import io
import os
import random
import re
import logging
from functools import partial
from pathlib import Path
from typing import Any

import accelerate
import datasets
import diffusers
import numpy as np
import ray.data
import torch
import transformers
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from PIL import Image
from torchvision.transforms import functional as TrF, InterpolationMode

from omni_diffusion.configs import (
    DataConfig, ModelConfig, OptimizerConfig, LoggingConfig, HubConfig
)

logger = get_logger(__name__, log_level="INFO")


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


class BaseTrainer(abc.ABC):
    def __init__(
        self,
        project_name: str = "omni-diffusion",
        output_dir: str = "./outputs",
        num_devices: int = 1,
        max_steps: int = 10000,
        train_batch_size: int = 16,
        validation_every_n_steps: int = 200,
        validation_prompt: str | None = None,
        num_validation_samples: int = 4,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        checkpointing_every_n_steps: int = 500,
        max_checkpoints: int | None = None,
        resume_from_checkpoint: str | None = None,
        # Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training.
        allow_tf32: bool = False,
        seed: int | None = None,
        gradient_clipping: float | None = 1.0,
        use_ema: bool = False,
        # The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`.
        # If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.
        prediction_type: str | None = None,
        mixed_precision: str | None = None,     # no, fp16, bf16
        use_xformers: bool = False,
        # The scale of noise offset.
        noise_offset: float = 0.0,
        # SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0.
        # More details here: https://arxiv.org/abs/2303.09556.
        snr_gamma: float | None = None,
        # When using SNR gamma with rescaled betas for zero terminal SNR, a divide-by-zero error can cause NaN
        # condition when computing the SNR with a sigma value of zero. This parameter overrides the check,
        # allowing the use of SNR gamma with a terminal SNR model. Use with caution, and closely monitor results.
        force_snr_gamma: bool = False,
        use_lora: bool = False,
        lora_rank: int = 4,
        train_text_encoder: bool = False,
        use_deepspeed: bool = False,
    ) -> None:
        self.project_name = project_name
        self.output_dir = os.path.realpath(output_dir)
        self.num_devices = num_devices
        self.max_steps = max_steps
        self.train_batch_size = train_batch_size
        self.validation_every_n_steps = validation_every_n_steps
        self.validation_prompt = validation_prompt
        self.num_validation_samples = num_validation_samples
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpointing_every_n_steps = checkpointing_every_n_steps
        self.max_checkpoints = max_checkpoints
        self.resume_from_checkpoint = resume_from_checkpoint
        self.allow_tf32 = allow_tf32
        self.seed = seed
        self.gradient_clipping = gradient_clipping
        self.use_ema = use_ema
        self.prediction_type = prediction_type
        self.mixed_precision = mixed_precision
        self.use_xformers = use_xformers
        self.noise_offset = noise_offset
        self.snr_gamma = snr_gamma
        self.force_snr_gamma = force_snr_gamma
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.train_text_encoder = train_text_encoder
        self.use_deepspeed = use_deepspeed

        # Data config
        self.dataset_name_or_path: str | None = None
        self.dataset_revision: str | None = None
        self.image_column: str | None = None
        self.prompt_columns: list[str] | None = None
        self.num_dataset_workers: int | None = None
        self.resolution: tuple[int, int] | None = None
        self.center_crop: bool | None = None
        self.random_flip: bool | None = None
        self.proportion_empty_prompts: float | None = None
        self.multi_aspect_training: bool | None = None

        # Model config
        self.model_name_or_path: str | None = None
        self.model_revision: str | None = None

        # Optimizer config
        self.learning_rate: float | None = None
        self.auto_scale_lr: bool | None = None
        self.use_8bit_adam: bool | None = None
        self.weight_decay: float | None = None
        self.adam_beta: tuple[float, float] | None = None
        self.adam_epsilon: float | None = None
        self.lr_scheduler: str | None = None
        self.lr_warmup_steps: int | None = None

        # Logging config
        self.logging_dir: str | None = None
        self.report_to: str | None = None

        # Hub config
        self.push_to_hub: bool | None = None
        self.hub_model_id: str | None = None
        self.commit_message: str | None = None
        self.private_hub: bool | None = None

        self.repo_id: str | None = None

        self.dataset_names: list[str] | None = None
        self.dataset_metas: dict[str, Any] | None = None

        self.configs = {
            "project_name": self.project_name,
            "output_dir": self.output_dir,
            "num_devices": self.num_devices,
            "max_steps": self.max_steps,
            "train_batch_size": self.train_batch_size,
            "validation_every_n_steps": self.validation_every_n_steps,
            "validation_prompt": self.validation_prompt,
            "num_validation_samples": self.num_validation_samples,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "checkpointing_every_n_steps": self.checkpointing_every_n_steps,
            "max_checkpoints": self.max_checkpoints,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "allow_tf32": self.allow_tf32,
            "seed": self.seed,
            "gradient_clipping": self.gradient_clipping,
            "use_ema": self.use_ema,
            "prediction_type": self.prediction_type,
            "mixed_precision": self.mixed_precision,
            "use_xformers": self.use_xformers,
            "noise_offset": self.noise_offset,
            "snr_gamma": self.snr_gamma,
            "force_snr_gamma": self.force_snr_gamma,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "train_text_encoder": self.train_text_encoder,
            "use_deepspeed": self.use_deepspeed
        }

    def prepare_configs(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        logging_config: LoggingConfig,
        hub_config: HubConfig,
    ) -> None:
        # Data config
        self.dataset_name_or_path = data_config.dataset_name_or_path
        self.dataset_revision = data_config.revision
        self.image_column = data_config.image_column
        if isinstance(data_config.prompt_column, str):
            prompt_columns = [data_config.prompt_column]
        else:
            prompt_columns = data_config.prompt_column
        self.prompt_columns = prompt_columns
        self.num_dataset_workers = data_config.num_workers
        if isinstance(data_config.resolution, int):
            resolution = (data_config.resolution, data_config.resolution)
        else:
            resolution = data_config.resolution
        self.resolution = resolution
        self.center_crop = data_config.center_crop
        self.random_flip = data_config.random_flip
        self.proportion_empty_prompts = data_config.proportion_empty_prompts
        self.multi_aspect_training = data_config.multi_aspect_training

        # Model config
        self.model_name_or_path = model_config.model_name_or_path
        self.model_revision = model_config.revision

        # Optimizer config
        self.learning_rate = optimizer_config.learning_rate
        self.auto_scale_lr = optimizer_config.auto_scale_lr
        self.use_8bit_adam = optimizer_config.use_8bit_adam
        self.weight_decay = optimizer_config.weight_decay
        self.adam_beta = optimizer_config.adam_beta
        self.adam_epsilon = optimizer_config.adam_epsilon
        self.lr_scheduler = optimizer_config.lr_scheduler
        self.lr_warmup_steps = optimizer_config.lr_warmup_steps

        # Logging config
        self.logging_dir = logging_config.logging_dir
        self.report_to = logging_config.report_to

        # Hub config
        self.push_to_hub = hub_config.push_to_hub
        self.hub_model_id = hub_config.model_id
        self.commit_message = hub_config.commit_message
        self.private_hub = hub_config.private

        self.configs.update({
            # Data config
            "dataset_name_or_path": self.dataset_name_or_path,
            "dataset_revision": self.dataset_revision,
            "image_column": self.image_column,
            "prompt_columns": self.prompt_columns,
            "num_dataset_workers": self.num_dataset_workers,
            "resolution": self.resolution,
            "center_crop": self.center_crop,
            "random_flip": self.random_flip,
            "proportion_empty_prompts": self.proportion_empty_prompts,
            "multi_aspect_training": self.multi_aspect_training,
            # Model config
            "model_name_or_path": self.model_name_or_path,
            "model_revision": self.model_revision,
            # Optimizer config
            "learning_rate": self.learning_rate,
            "auto_scale_lr": self.auto_scale_lr,
            "use_8bit_adam": self.use_8bit_adam,
            "weight_decay": self.weight_decay,
            "adam_beta": self.adam_beta,
            "adam_epsilon": self.adam_epsilon,
            "lr_scheduler": self.lr_scheduler,
            "lr_warmup_steps": self.lr_warmup_steps,
            # Logging config
            "logging_dir": self.logging_dir,
            "report_to": self.report_to,
            # Hub config
            "push_to_hub": self.push_to_hub,
            "hub_model_id": self.hub_model_id,
            "commit_message": self.commit_message,
            "private_hub": self.private_hub,
        })

    def prepare_data(self) -> dict[str, ray.data.Dataset]:
        # TODO: support s3 paths

        sub_paths: list[Path] = []
        if self.multi_aspect_training:
            for sub_path in Path(self.dataset_name_or_path).glob("*_*"):
                if not sub_path.is_dir():
                    continue

                sub_paths.append(sub_path.resolve())
        else:
            sub_paths.append(Path(self.dataset_name_or_path).resolve())

        columns = [self.image_column] + self.prompt_columns

        datasets = {}
        dataset_metas = {}
        for sub_path in sub_paths:
            match = re.match("^([0-9]+)_([0-9]+)_([0-9]+)$", sub_path.stem)
            if match:
                num_samples = int(match.group(3))
                target_size = (int(match.group(1)), int(match.group(2)))

                # TODO: fix me
                if num_samples < 2048 * 1000:
                    continue
            else:
                num_samples = 1
                target_size = None

            if self.resolution is not None:
                if isinstance(self.resolution, int):
                    target_size = (self.resolution, self.resolution)
                else:
                    target_size = self.resolution

            ds = ray.data.read_parquet(sub_path, columns=columns)

            # TODO: randomize block order
            # ds = ds.randomize_block_order(seed=self.seed)

            ds = ds.map(partial(
                select_prompt,
                prompt_columns=self.prompt_columns,
                proportion_empty_prompts=self.proportion_empty_prompts,
            ))

            ds = ds.drop_columns([p for p in self.prompt_columns if p != "prompt"])

            ds = self.apply_tokenizer(ds)

            ds = ds.drop_columns(["prompt"])

            ds = ds.map(partial(
                process_image,
                target_size=target_size,
                center_crop=self.center_crop,
                random_flip=self.random_flip,
            ))

            datasets[sub_path.stem] = ds
            dataset_metas[sub_path.stem] = {
                "num_samples": num_samples,
                "target_size": target_size,
            }

        self.dataset_names = list(datasets.keys())
        self.dataset_metas = dataset_metas

        return datasets

    def apply_tokenizer(self, ds: ray.data.Dataset) -> ray.data.Dataset:
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self):
        ...

    @abc.abstractmethod
    def validate(self):
        ...

    @abc.abstractmethod
    def test(self):
        ...

    @abc.abstractmethod
    def predict(self):
        ...

    def setup_accelerator(self) -> Accelerator:
        # Train has a bug somewhere that causes ACCELERATE_TORCH_DEVICE to not be set
        # properly on multi-gpu nodes
        cuda_visible_device = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        local_rank = int(os.environ["LOCAL_RANK"])
        device_id = cuda_visible_device[local_rank]
        os.environ["ACCELERATE_TORCH_DEVICE"] = f"cuda:{device_id}"

        logging_dir = os.path.join(self.output_dir, self.logging_dir)

        project_config = ProjectConfiguration(
            project_dir=self.output_dir,
            logging_dir=logging_dir,
        )

        if self.use_deepspeed:
            deepspeed_plugin = DeepSpeedPlugin(
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                gradient_clipping=self.gradient_clipping,
                zero_stage=2,
                offload_optimizer_device="none",
                offload_param_device="none",
            )
            deepspeed_plugin.hf_ds_config.config["train_micro_batch_size_per_gpu"] = self.train_batch_size
        else:
            deepspeed_plugin = None

        accelerator = Accelerator(
            deepspeed_plugin=deepspeed_plugin,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            project_config=project_config,
            log_with=self.report_to,
        )

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if self.seed is not None:
            set_seed(self.seed)

        return accelerator

    def setup_optimizer(
        self,
        parameters,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        if self.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "You need to install the bitsandbytes package to use 8-bit AdamW: `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            parameters,
            lr=self.learning_rate,
            betas=self.adam_beta,
            weight_decay=self.weight_decay,
            eps=self.adam_epsilon,
        )

        lr_scheduler = get_scheduler(
            self.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps * self.gradient_accumulation_steps,
            num_training_steps=self.max_steps * self.gradient_accumulation_steps,
        )

        return optimizer, lr_scheduler


class MultiAspectDataLoader:
    def __init__(self, datasets: dict[str, ray.data.Dataset]) -> None:
        self.datasets = datasets


def select_prompt(
    row: dict[str, Any],
    prompt_columns: list[str],
    proportion_empty_prompts: float = 0.0,
) -> dict[str, Any]:
    if random.random() < proportion_empty_prompts:
        row["prompt"] = ""
        return row

    prompts = []

    for prompt_column in prompt_columns:
        prompt = row[prompt_column]
        if isinstance(prompt, str):
            prompts.append(prompt)
        elif isinstance(prompt, (list, np.ndarray)):
            prompts.extend(list(prompt))
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

    assert len(prompts) > 0, prompts

    row["prompt"] = random.choice(prompts)

    return row


def process_image(
    row: dict[str, Any],
    target_size: tuple[int, int] | None = None,
    center_crop: bool = False,
    random_flip: bool = False,
) -> dict[str, Any]:
    image: Image.Image = Image.open(io.BytesIO(row["image"]))

    original_size = (image.height, image.width)
    if target_size is None:
        target_size = original_size

    image: Image.Image = TrF.resize(
        image,
        size=min(*target_size),
        interpolation=InterpolationMode.LANCZOS,
    )

    if center_crop:
        y1 = max(0, int(round(image.height - target_size[0]) / 2.0))
        x1 = max(0, int(round(image.width - target_size[1]) / 2.0))
    else:
        y1 = random.randint(0, max(0, image.height - target_size[0]))
        x1 = random.randint(0, max(0, image.width - target_size[1]))

    image = TrF.crop(image, y1, x1, target_size[0], target_size[1])

    if random_flip and random.random() < 0.5:
        x1 = image.width - x1 - target_size[1]
        image = TrF.hflip(image)

    image: torch.Tensor = TrF.to_tensor(image)
    image = TrF.normalize(image, mean=[0.5], std=[0.5])

    row["image"] = image.numpy()
    row["original_size"] = np.array(original_size, dtype=np.int64)
    row["crop_top_left"] = np.array((y1, x1), dtype=np.int64)
    row["target_size"] = np.array(target_size, dtype=np.int64)

    return row
