import abc
import os
import logging

import accelerate
import datasets
import diffusers
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler

from omni_diffusion.configs import (
    DataConfig, ModelConfig, OptimizerConfig
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
        max_epochs: int = 1,
        max_steps: int | None = None,
        train_batch_size: int = 16,
        validation_every_n_steps: int = 200,
        validation_prompt: str | None = None,
        num_validation_samples: int = 4,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        checkpointing_every_n_steps: int = 500,
        max_checkpoints: int | None = None,
        resume_from_checkpoint: str | None = None,
        project_name: str = "omni-diffusion",
        output_dir: str = "./outputs",
        logging_dir: str = "logs",
        allow_tf32: bool = False,
        seed: int | None = None,
        max_grad_norm: float = 1.0,
        use_ema: bool = False,
        prediction_type: str | None = None,
        mixed_precision: str | None = None,
        use_xformers: bool = False,
        noise_offset: float = 0.0,
        proportion_empty_prompts: float = 0.0,
        snr_gamma: float | None = None,
        force_snr_gamma: bool = False,
        push_to_hub: bool = False,
        hub_model_id: str | None = None,
        report_to: str = "tensorboard",
        use_lora: bool = False,
        lora_rank: int = 4,
        train_text_encoder: bool = False,
    ) -> None:
        self.max_epochs = max_epochs
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
        self.project_name = project_name
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.allow_tf32 = allow_tf32
        self.seed = seed
        self.max_grad_norm = max_grad_norm
        self.use_ema = use_ema
        self.prediction_type = prediction_type
        self.mixed_precision = mixed_precision
        self.use_xformers = use_xformers
        self.noise_offset = noise_offset
        self.proportion_empty_prompts = proportion_empty_prompts
        self.snr_gamma = snr_gamma
        self.force_snr_gamma = force_snr_gamma
        self.push_to_hub = push_to_hub
        self.hub_model_id = hub_model_id
        self.report_to = report_to
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.train_text_encoder = train_text_encoder

        self.config = {
            "max_epochs": self.max_epochs,
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
            "project_name": self.project_name,
            "output_dir": self.output_dir,
            "logging_dir": self.logging_dir,
            "allow_tf32": self.allow_tf32,
            "seed": self.seed,
            "max_grad_norm": self.max_grad_norm,
            "use_ema": self.use_ema,
            "prediction_type": self.prediction_type,
            "mixed_precision": self.mixed_precision,
            "use_xformers": self.use_xformers,
            "noise_offset": self.noise_offset,
            "proportion_empty_prompts": self.proportion_empty_prompts,
            "snr_gamma": self.snr_gamma,
            "force_snr_gamma": self.force_snr_gamma,
            "push_to_hub": self.push_to_hub,
            "hub_model_id": self.hub_model_id,
            "report_to": self.report_to,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "train_text_encoder": self.train_text_encoder,
        }

    def fit(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
    ):
        ...

    def validate(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
    ):
        ...

    def test(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
    ):
        ...

    def predict(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
    ):
        ...

    def setup_accelerator(self):
        logging_dir = os.path.join(self.output_dir, self.logging_dir)

        project_config = ProjectConfiguration(
            project_dir=self.output_dir,
            logging_dir=logging_dir,
        )

        accelerator = Accelerator(
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
        optimizer_config: OptimizerConfig,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        if optimizer_config.use_8bit_adam:
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
            lr=optimizer_config.learning_rate,
            betas=optimizer_config.adam_beta,
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.adam_epsilon,
        )

        lr_scheduler = get_scheduler(
            optimizer_config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=optimizer_config.lr_warmup_steps * self.gradient_accumulation_steps,
            num_training_steps=self.max_steps * self.gradient_accumulation_steps,
        )

        return optimizer, lr_scheduler
