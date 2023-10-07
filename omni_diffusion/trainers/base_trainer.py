import abc
import io
import itertools
import os
import random
import re
import logging
import shutil
from functools import partial
from pathlib import Path
from typing import Any

import accelerate
import datasets
import diffusers
import numpy as np
import ray.data
import ray.train
import torch
import transformers
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EMAModel,
    UNet2DConditionModel,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import upload_folder
from packaging import version
from PIL import Image
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.utils import ContextManagers
from tqdm import tqdm
from torch.nn import functional as F
from torchvision.transforms import functional as TrF, InterpolationMode

from omni_diffusion.configs import (
    DataConfig, ModelConfig, OptimizerConfig, LoggingConfig, HubConfig
)
from omni_diffusion.utils.multi_source_dataloader import MultiSourceDataLoader

logger = get_logger(__name__, log_level="INFO")


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


def import_model_class_from_model_name_or_path(
    model_name_or_path: str,
    revision: str | None = None,
    subfolder: str | None = None,
) -> PreTrainedModel:
    text_encoder_config = PretrainedConfig.from_pretrained(
        model_name_or_path,
        revision=revision,
        subfolder=subfolder,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"Unknown text encoder class: {model_class}")


def unet_attn_processors_state_dict(unet: UNet2DConditionModel) -> dict[str, torch.Tensor]:
    """
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[f"{attn_processor_key}.{parameter_key}"] = parameter

    return attn_processors_state_dict


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
        pipeline_cls: str = "StableDiffusionPipeline"
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

        if pipeline_cls == "StableDiffusionPipeline":
            self.pipeline_cls = StableDiffusionPipeline
            self.num_text_encoders = 1
        elif pipeline_cls == "StableDiffusionXLPipeline":
            self.pipeline_cls = StableDiffusionXLPipeline
            self.num_text_encoders = 2
        else:
            raise ValueError(f"Unknown pipeline_cls: {pipeline_cls}")

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
        self.local_shuffle_buffer_size: int | None = None
        self.prefetch_batches: int | None = None

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

        if use_lora and use_ema:
            raise ValueError("LoRA and EMA cannot be used together.")

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
            "use_deepspeed": self.use_deepspeed,
            "pipeline_cls": pipeline_cls,
            "num_text_encoders": self.num_text_encoders,
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
        self.local_shuffle_buffer_size = data_config.local_shuffle_buffer_size
        self.prefetch_batches = data_config.prefetch_batches

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
            "local_shuffle_buffer_size": self.local_shuffle_buffer_size,
            "prefetch_batches": self.prefetch_batches,
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
        ds = ds.map_batches(
            TokenizerActor,
            fn_constructor_kwargs={
                "model_name_or_path": self.model_name_or_path,
                "revision": self.model_revision,
                "num_tokenizers": self.num_text_encoders,
            },
            compute=ray.data.ActorPoolStrategy(size=self.num_dataset_workers),
        )

        return ds

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

    def setup_dataloader(self) -> MultiSourceDataLoader:
        train_dataloader = MultiSourceDataLoader(
            batch_size=self.train_batch_size,
            local_shuffle_buffer_size=self.local_shuffle_buffer_size,
            local_shuffle_seed=self.seed,
            prefetch_batches=self.prefetch_batches,
        )
        for dataset_name in self.dataset_names:
            meta = self.dataset_metas[dataset_name]
            ds = ray.train.get_dataset_shard(dataset_name)
            train_dataloader.add_dataset(ds, weight=meta["num_samples"])

        return train_dataloader

    def setup_optimizer(
        self,
        parameters: list[torch.nn.Parameter],
        accelerator: Accelerator,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        if self.auto_scale_lr:
            self.learning_rate = (
                self.learning_rate * self.gradient_accumulation_steps * self.train_batch_size * accelerator.num_processes
            )

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

    def setup_noise_scheduler(self) -> DDPMScheduler:
        # Load scheduler and models.
        noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            self.model_name_or_path,
            revision=self.model_revision,
            subfolder="scheduler",
        )

        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            noise_scheduler.register_to_config(prediction_type=self.prediction_type)

        # Check for terminal SNR in combination with SNR Gamma
        if (
            self.snr_gamma is not None
            and not self.force_snr_gamma
            and (
                hasattr(noise_scheduler.config, "rescale_betas_zero_snr") and noise_scheduler.config.rescale_betas_zero_snr
            )
        ):
            raise ValueError(
                f"The selected noise scheduler for the model `{self.model_name_or_path}` uses rescaled betas for zero SNR.\n"
                "When this configuration is present, the parameter `snr_gamma` may not be used without parameter `force_snr_gamma`.\n"
                "This is due to a mathematical incompatibility between our current SNR gamma implementation, and a sigma value of zero."
            )

        return noise_scheduler

    def setup_models(
            self, accelerator: Accelerator, num_text_encoders: int = 1
    ) -> tuple[list[PreTrainedModel], AutoencoderKL, UNet2DConditionModel, EMAModel | None, list[torch.nn.Parameter]]:
        assert num_text_encoders in [1, 2], num_text_encoders

        text_encode_cls_one = import_model_class_from_model_name_or_path(
            self.model_name_or_path,
            revision=self.model_revision,
            subfolder="text_encoder",
        )

        # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
        # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
        # will try to assign the same optimizer with the same weights to all models during
        # `deepspeed.initialize`, which of course doesn't work.
        #
        # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
        # frozen models from being partitioned during `zero.Init` which gets called during
        # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
        # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            text_encoder_one = text_encode_cls_one.from_pretrained(
                self.model_name_or_path,
                revision=self.model_revision,
                subfolder="text_encoder",
            )

        if num_text_encoders == 2:
            text_encode_cls_two = import_model_class_from_model_name_or_path(
                self.model_name_or_path,
                revision=self.model_revision,
                subfolder="text_encoder_2",
            )

            text_encoder_two = text_encode_cls_two.from_pretrained(
                self.model_name_or_path,
                revision=self.model_revision,
                subfolder="text_encoder_2",
            )
        else:
            text_encoder_two = None

        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            vae: AutoencoderKL = AutoencoderKL.from_pretrained(
                self.model_name_or_path,
                revision=self.model_revision,
                subfolder="vae",
            )

        unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            self.model_name_or_path,
            revision=self.model_revision,
            subfolder="unet",
        )

        # Freeze vae and text encoders.
        vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        if text_encoder_two:
            text_encoder_two.requires_grad_(False)

        if self.use_lora:
            # We only train the additional adapter LoRA layers.
            unet.requires_grad_(False)

        unet.train()
        vae.eval()
        text_encoder_one.eval()

        if text_encoder_two:
            text_encoder_two.eval()

        # For mixed precision training we cast all non-trainable weigths to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        if self.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32

        # The VAE is in full precision to avoid NaN losses.
        vae.to(accelerator.device, dtype=torch.float32)
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        if text_encoder_two:
            text_encoder_two.to(accelerator.device, dtype=weight_dtype)

        if self.use_lora:
            unet.to(accelerator.device, dtype=weight_dtype)

        if self.use_ema:
            assert not self.use_lora, "EMA is not supported with LoRA"

            ema_unet_base: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
                self.model_name_or_path,
                revision=self.model_revision,
                subfolder="unet",
            )
            ema_unet = EMAModel(
                ema_unet_base.parameters(),
                model_cls=UNet2DConditionModel,
                model_config=ema_unet_base.config,
            )
        else:
            ema_unet = None

        if self.use_xformers:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. "
                        "If you observe problems during training, please update xFormers to at least 0.0.17. "
                        "See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if self.use_lora:
            # now we will add new LoRA weights to the attention layers
            # Set correct lora layers
            unet_lora_attn_procs = {}
            unet_lora_parameters = []
            for name, attn_processor in unet.attn_processors.items():
                cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]

                lora_attn_processor_class = (
                    LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
                )
                module = lora_attn_processor_class(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=self.lora_rank,
                )
                unet_lora_attn_procs[name] = module
                unet_lora_parameters.extend(module.parameters())

            unet.set_attn_processor(unet_lora_attn_procs)

            if self.train_text_encoder:
                # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
                text_lora_parameters_one = LoraLoaderMixin._modify_text_encoder(
                    text_encoder_one, dtype=torch.float32, rank=self.lora_rank
                )

                if text_encoder_two:
                    text_lora_parameters_two = LoraLoaderMixin._modify_text_encoder(
                        text_encoder_two, dtype=torch.float32, rank=self.lora_rank
                    )
                else:
                    text_lora_parameters_two = []

            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                # there are only two options here. Either are just the unet attn processor layers
                # or there are the unet and text encoder atten layers
                unet_lora_layers_to_save = None
                text_encoder_one_lora_layers_to_save = None
                text_encoder_two_lora_layers_to_save = None

                unet_type = type(accelerator.unwrap_model(unet))
                text_encoder_one_type = type(accelerator.unwrap_model(text_encoder_one))
                if text_encoder_two:
                    text_encoder_two_type = type(accelerator.unwrap_model(text_encoder_two))

                for model in models:
                    if isinstance(model, unet_type):
                        unet_lora_layers_to_save = unet_attn_processors_state_dict(model)
                    elif isinstance(model, text_encoder_one_type):
                        text_encoder_one_lora_layers_to_save = text_encoder_lora_state_dict(model)
                    elif text_encoder_two and isinstance(model, text_encoder_two_type):
                        text_encoder_two_lora_layers_to_save = text_encoder_lora_state_dict(model)
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if len(weights) > 0:
                        weights.pop()

                StableDiffusionXLPipeline.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                    text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
                )

            def load_model_hook(models, input_dir):
                unet_ = None
                text_encoder_one_ = None
                text_encoder_two_ = None

                unet_type = type(accelerator.unwrap_model(unet))
                text_encoder_one_type = type(accelerator.unwrap_model(text_encoder_one))
                if text_encoder_two:
                    text_encoder_two_type = type(accelerator.unwrap_model(text_encoder_two))

                while len(models) > 0:
                    model = models.pop()

                    if isinstance(model, unet_type):
                        unet_ = model
                    elif isinstance(model, text_encoder_one_type):
                        text_encoder_one_ = model
                    elif text_encoder_two and isinstance(model, text_encoder_two_type):
                        text_encoder_two_ = model
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
                LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)

                text_encoder_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder." in k}
                LoraLoaderMixin.load_lora_into_text_encoder(
                    text_encoder_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_one_
                )

                if text_encoder_two:
                    text_encoder_2_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder_2." in k}
                    LoraLoaderMixin.load_lora_into_text_encoder(
                        text_encoder_2_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_two_
                    )
        else:
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(
                models: list[UNet2DConditionModel],
                weights: list[torch.Tensor],
                output_dir: str,
            ):
                if self.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for model in models:
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if len(weights) > 0:
                        weights.pop()

            def load_model_hook(models: list[UNet2DConditionModel], input_dir: str):
                if self.use_ema:
                    loaded_model = EMAModel.from_pretrained(
                        os.path.join(input_dir, "unet_ema"), UNet2DConditionModel
                    )
                    ema_unet.load_state_dict(loaded_model.state_dict())
                    ema_unet.to(accelerator.device)
                    del loaded_model

                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    loaded_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**loaded_model.config)

                    model.load_state_dict(loaded_model.state_dict())
                    model.to(accelerator.device)
                    del loaded_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        if self.gradient_checkpointing:
            assert not self.use_lora, "Gradient checkpointing is not supported with LoRA"

            unet.enable_gradient_checkpointing()

        if self.use_lora:
            params_to_optimize = unet_lora_parameters

            if self.train_text_encoder:
                params_to_optimize = itertools.chain(
                    params_to_optimize, text_lora_parameters_one, text_lora_parameters_two
                )
        else:
            params_to_optimize = unet.parameters()

        text_encoders = [text_encoder_one]
        if text_encoder_two:
            text_encoders.append(text_encoder_two)

        return text_encoders, vae, unet, ema_unet, params_to_optimize

    def fit(self):
        if self.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        accelerator = self.setup_accelerator()

        train_dataloader = self.setup_dataloader()

        noise_scheduler = self.setup_noise_scheduler()

        text_encoders, vae, unet, ema_unet, params_to_optimize = self.setup_models(
            accelerator=accelerator,
            num_text_encoders=self.num_text_encoders,
        )

        optimizer, lr_scheduler = self.setup_optimizer(params_to_optimize, accelerator=accelerator)

        unet, optimizer, lr_scheduler = accelerator.prepare(
            unet, optimizer, lr_scheduler
        )

        if self.train_text_encoder:
            text_encoders = [
                accelerator.prepare(text_encoder)
                for text_encoder in text_encoders
            ]

        if accelerator.is_main_process:
            accelerator.init_trackers(
                self.project_name,
                config=self.configs,
            )

        if self.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32

        total_batch_size = self.train_batch_size * accelerator.num_processes * self.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_steps}")
        global_step = 0

        # Load in the weights and states from a previous checkpoint.
        if self.resume_from_checkpoint is not None:
            if self.resume_from_checkpoint != "latest":
                dir_name = os.path.basename(self.resume_from_checkpoint)
            else:
                # Get the latest checkpoint in the output dir.
                dirs = os.listdir(self.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                dir_name = dirs[-1] if len(dirs) > 0 else None

            if dir_name is None:
                logger.warning(
                    f"Provided `resume_from_checkpoint` ({self.resume_from_checkpoint}) does not exist. "
                    "Training from scratch."
                )
                self.resume_from_checkpoint = None
            else:
                logger.info(f"Loading model from {dir_name}")
                accelerator.load_state(os.path.join(self.output_dir, dir_name))
                global_step = int(dir_name.split("-")[1])

                resume_step = global_step * self.gradient_accumulation_steps

        progress_bar = tqdm(
            range(0, self.max_steps),
            desc="Steps",
            disable=not accelerator.is_local_main_process,
        )

        for epoch in range(10000000):
            unet.train()
            if self.train_text_encoder:
                for text_encoder in text_encoders:
                    text_encoder.train()

            train_loss = 0.0

            # TODO: keep aspec ratio within a batch
            # TODO: quick skip or random block order
            for step, batch in enumerate(train_dataloader):
                if self.resume_from_checkpoint is not None and epoch == 0 and step < resume_step:
                    if step % self.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with accelerator.accumulate(unet):
                    loss = self.training_step(
                        batch,
                        unet=unet,
                        vae=vae,
                        text_encoders=text_encoders,
                        noise_scheduler=noise_scheduler,
                    )

                    # Gather the losses across all processes for logging (if we use distributed training).
                    losses = accelerator.gather(loss.repeat(self.train_batch_size))
                    train_loss += losses.mean().item()

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_to_optimize, self.gradient_clipping)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if self.use_ema:
                        ema_unet.step(unet.parameters())

                    progress_bar.update(1)
                    global_step += 1
                    train_loss = train_loss / self.gradient_accumulation_steps
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % self.checkpointing_every_n_steps == 0:
                        if accelerator.is_main_process:
                            if self.max_checkpoints is not None:
                                checkpoints = os.listdir(self.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `max_checkpoints - 1` checkpoints
                                if len(checkpoints) >= self.max_checkpoints:
                                    num_to_remove = len(checkpoints) - self.max_checkpoints + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, "
                                        f"removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(self.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                        accelerator.wait_for_everyone()

                        save_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if accelerator.is_main_process:
                        if global_step % self.validation_every_n_steps == 0:
                            self.validation_step(
                                global_step,
                                accelerator=accelerator,
                                model_name_or_path=self.model_name_or_path,
                                model_revision=self.model_revision,
                                text_encoders=text_encoders,
                                vae=vae,
                                unet=unet,
                                ema_unet=ema_unet,
                                weight_dtype=weight_dtype,
                            )

                if global_step >= self.max_steps:
                    break

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            unet = accelerator.unwrap_model(unet)
            if self.use_ema:
                ema_unet.copy_to(unet.parameters())

            if self.train_text_encoder:
                text_encoder_lora_layers_kwargs = {}
                for i, text_encoder in enumerate(text_encoders):
                    text_encoder = accelerator.unwrap_model(text_encoder)
                    text_encoder_lora_layers = text_encoder_lora_state_dict(text_encoder)

                    name = f"text_encoder_{i + 1}_lora_layers" if i > 0 else "text_encoder_lora_layers"
                    text_encoder_lora_layers_kwargs[name] = text_encoder_lora_layers
            else:
                text_encoder_lora_layers_kwargs = {}

            if self.use_lora:
                unet_lora_layers = unet_attn_processors_state_dict(unet)


                self.pipeline_cls.save_lora_weights(
                    save_directory=self.output_dir,
                    unet_lora_layers=unet_lora_layers,
                    **text_encoder_lora_layers_kwargs,
                )

                del unet
                del text_encoders
                del text_encoder_lora_layers_kwargs
                torch.cuda.empty_cache()

                # Final inference
                # Load previous pipeline
                pipeline = self.pipeline_cls.from_pretrained(
                    self.model_name_or_path,
                    revision=self.model_revision,
                    vae=vae,
                    safty_checker=None,
                    torch_dtype=weight_dtype,
                )
                pipeline = pipeline.to(accelerator.device)

                # Load attention processors
                pipeline.load_lora_weights(self.output_dir)
            else:
                # Serialize pipeline.
                pipeline = self.pipeline_cls.from_pretrained(
                    self.model_name_or_path,
                    revision=self.model_revision,
                    unet=unet,
                    vae=vae,
                    safty_checker=None,
                    torch_dtype=weight_dtype,
                )
                if self.prediction_type is not None:
                    scheduler_args = {"prediction_type": self.prediction_type}
                    pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
                pipeline.save_pretrained(self.output_dir)

            # run inference
            pipeline = pipeline.to(accelerator.device)
            images = self.inference_step(global_step, accelerator, pipeline, stage="test")

            if self.push_to_hub:
                self.save_model_card(
                    repo_id=self.repo_id,
                    images=images,
                    validation_prompt=self.validation_prompt,
                    base_model=self.model_name_or_path,
                    dataset_name=self.dataset_name_or_path,
                    repo_folder=self.output_dir,
                    vae_path=self.model_name_or_path,
                )
                upload_folder(
                    repo_id=self.repo_id,
                    folder_path=self.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        accelerator.end_training()

    @abc.abstractmethod
    def training_step(
        self,
        batch,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        text_encoders: list[PreTrainedModel],
        noise_scheduler: DDPMScheduler,
    ) -> torch.Tensor:
        ...

    def validation_step(
        self,
        current_step: int,
        accelerator: Accelerator,
        model_name_or_path: str,
        model_revision: str | None,
        text_encoders: list[PreTrainedModel],
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        ema_unet: EMAModel,
        weight_dtype: torch.dtype,
    ):
        if not accelerator.is_main_process or not self.validation_prompt or self.num_validation_samples == 0:
            return

        logger.info(
            f"Running validation... \n Generating {self.num_validation_samples} images with prompt:"
            f" {self.validation_prompt}."
        )
        if self.use_ema:
            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
            ema_unet.store(unet.parameters())
            ema_unet.copy_to(unet.parameters())

        text_encoders_kwargs = {}
        for i, text_encoder in enumerate(text_encoders):
            text_encoder = accelerator.unwrap_model(text_encoder)
            name = f"text_encoder_{i + 1}" if i > 0 else "text_encoder"
            text_encoders_kwargs[name] = text_encoder

        # create pipeline
        pipeline = self.pipeline_cls.from_pretrained(
            model_name_or_path,
            revision=model_revision,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            safety_checker=None,
            torch_dtype=weight_dtype,
            **text_encoders_kwargs,
        )
        if self.prediction_type is not None:
            scheduler_args = {"prediction_type": self.prediction_type}
            pipeline.scheduler = pipeline.scheduler.from_config(
                pipeline.scheduler.config,
                **scheduler_args,
            )

        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        self.inference_step(current_step, accelerator, pipeline, stage="validation")

        del pipeline
        torch.cuda.empty_cache()

        ema_unet.restore(unet.parameters())

    def inference_step(
        self,
        current_step: int,
        accelerator: Accelerator,
        pipeline: StableDiffusionPipeline,
        stage: str = "validation",
    ):
        if not accelerator.is_main_process or not self.validation_prompt or self.num_validation_samples == 0:
            return []

        generator = torch.Generator(device=accelerator.device).manual_seed(self.seed) if self.seed else None
        pipeline_args = {"prompt": self.validation_prompt}

        with torch.cuda.amp.autocast():
            images = [
                pipeline(**pipeline_args, generator=generator).images[0]
                for _ in range(self.num_validation_samples)
            ]

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images(stage, np_images, current_step, dataformats="NHWC")
            if tracker.name == "wandb":
                import wandb

                tracker.log(
                    {
                        stage: [
                            wandb.Image(image, caption=f"{i}: {self.validation_prompt}")
                            for i, image in enumerate(images)
                        ]
                    }
                )

        return images

    def save_model_card(
        self,
        repo_id: str,
        images=None,
        validation_prompt=None,
        base_model=str,
        dataset_name=str,
        repo_folder=None,
        vae_path=None,
    ) -> None:
        ...


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


class TokenizerActor:
    def __init__(
        self,
        model_name_or_path: str,
        revision: str | None = None,
        num_tokenizers: int = 1,
    ):
        super().__init__()

        assert num_tokenizers in [1, 2]

        self.tokenizer_one = AutoTokenizer.from_pretrained(
            model_name_or_path,
            revision=revision,
            subfolder="tokenizer",
            use_fast=True,
        )

        if num_tokenizers > 1:
            self.tokenizer_two = AutoTokenizer.from_pretrained(
                model_name_or_path,
                revision=revision,
                subfolder="tokenizer_2",
                use_fast=True,
            )
        else:
            self.tokenizer_two = None

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        prompts = list(batch["prompt"])

        input_ids_one = self.tokenizer_one(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_one.model_max_length,
            return_tensors="np",
        ).input_ids

        if self.tokenizer_two is None:
            batch["input_ids"] = input_ids_one
        else:
            input_ids_two = self.tokenizer_two(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer_two.model_max_length,
                return_tensors="np",
            ).input_ids

            batch["input_ids_one"] = input_ids_one
            batch["input_ids_two"] = input_ids_two

        return batch


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


@torch.jit.script
def compute_snr(
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    # TODO: check if `sqrt_alphas_cumprod` is on the right device
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def compute_vae_encodings(batch, vae: AutoencoderKL) -> torch.Tensor:
    pixel_values = batch.pop("image")

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    return model_input * vae.config.scaling_factor


def encode_prompts_sd(
    batch,
    text_encoders: list[PreTrainedModel],
    train_text_encoder: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(text_encoders) == 1, "Only one text encoder is supported for now"

    with torch.set_grad_enabled(train_text_encoder):
        return text_encoders[0](input_ids=batch["input_ids"].to(text_encoders[0].device))[0]


def encode_prompts_sdxl(
    batch,
    text_encoders: list[PreTrainedModel],
    train_text_encoder: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_embeds_list = []

    input_ids_list = [batch["input_ids_one"], batch["input_ids_two"]]

    with torch.set_grad_enabled(train_text_encoder):
        for input_ids, text_encoder in zip(input_ids_list, text_encoders):
            prompt_embeds = text_encoder(
                input_ids=input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

    return prompt_embeds, pooled_prompt_embeds
