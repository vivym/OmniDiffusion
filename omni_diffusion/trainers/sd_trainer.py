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
            text_encoder = CLIPTextModel.from_pretrained(
                model_config.model_name_or_path,
                subfolder="text_encoder",
                revision=model_config.revision,
            )
            vae = AutoencoderKL.from_pretrained(
                model_config.model_name_or_path,
                subfolder="vae",
                revision=model_config.revision,
            )

        unet = UNet2DConditionModel.from_pretrained(
            model_config.model_name_or_path,
            subfolder="unet",
            revision=model_config.non_ema_revision,
        )

    def validate(self):
        ...

    def test(self):
        ...

    def predict(self):
        ...
