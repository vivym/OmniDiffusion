from accelerate.state import AcceleratorState
from diffusers import DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from omni_diffusion.configs import (
    DataConfig, ModelConfig, OptimizerConfig
)
from .base_trainer import BaseTrainer, deepspeed_zero_init_disabled_context_manager


class StableDiffusionTrainer(BaseTrainer):
    def fit(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
    ):
        accelerator = self.setup_accelerator()

        # Load scheduler, tokenizer and models.
        noise_scheduler = DDPMScheduler.from_pretrained(
            model_config.model_name_or_path,
            subfolder="scheduler",
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            model_config.model_name_or_path,
            subfolder="tokenizer",
            revision=model_config.revision,
        )

    def validate(self):
        ...

    def test(self):
        ...

    def predict(self):
        ...
