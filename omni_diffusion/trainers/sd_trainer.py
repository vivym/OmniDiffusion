import torch
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from torch.nn import functional as F
from transformers import PreTrainedModel

from .base_trainer import BaseTrainer, compute_snr, compute_vae_encodings, encode_prompts_sd

logger = get_logger(__name__, log_level="INFO")


class StableDiffusionTrainer(BaseTrainer):
    def training_step(
        self,
        batch,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        text_encoders: list[PreTrainedModel],
        noise_scheduler: DDPMScheduler,
    ) -> torch.Tensor:
        prompt_embeds = encode_prompts_sd(
            batch,
            text_encoders=text_encoders,
            train_text_encoder=self.train_text_encoder,
        )

        model_input = compute_vae_encodings(batch, vae)
        noise = torch.randn_like(model_input)
        if self.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.noise_offset * torch.randn(
                (model_input.shape[0], model_input.shape[1], 1, 1),
                device=model_input.device,
            )

        bsz = model_input.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,),
            dtype=torch.int64,
            device=model_input.device,
        )

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

        model_pred = unet(noisy_model_input, timesteps, prompt_embeds).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(timesteps, noise_scheduler.alphas_cumprod)
            if noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss
