import abc
import os
import logging
from pathlib import Path

import accelerate
import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder

from omni_diffusion.configs import (
    DataConfig, ModelConfig, OptimizerConfig
)

logger = logging.getLogger(__name__)


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
        output_dir: str = "./outputs",
        logging_dir: str = "logs",
        allow_tf32: bool = False,
        seed: int | None = None,
        max_grad_norm: float = 1.0,
        use_ema: bool = False,
        prediction_type: str | None = None,
        mixed_precision: bool = False,
        noise_offset: float = 0.0,
        proportion_empty_prompts: float = 0.0,
        snr_gamma: float | None = None,
        force_snr_gamma: bool = False,
        push_to_hub: bool = False,
        hub_model_id: str | None = None,
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
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.allow_tf32 = allow_tf32
        self.seed = seed
        self.max_grad_norm = max_grad_norm
        self.use_ema = use_ema
        self.prediction_type = prediction_type
        self.mixed_precision = mixed_precision
        self.noise_offset = noise_offset
        self.proportion_empty_prompts = proportion_empty_prompts
        self.snr_gamma = snr_gamma
        self.force_snr_gamma = force_snr_gamma
        self.push_to_hub = push_to_hub
        self.hub_model_id = hub_model_id

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
            log_with="tensorboard", # TODO: support more loggers
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

        if accelerator.is_main_process:
            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)
            
            if self.push_to_hub:
                repo_id = create_repo(
                    repo_id=self.hub_model_id or Path(self.output_dir).name,
                    exist_ok=True,
                ).repo_id

        accelerator.wait_for_everyone()

        return accelerator
