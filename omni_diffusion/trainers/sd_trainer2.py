import torch
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from omni_diffusion.data.streaming_dataset import MMStreamingDataset
from .base_trainer import BaseTrainer, compute_snr, compute_vae_encodings, encode_prompts_sd

logger = get_logger(__name__, log_level="INFO")


class StableDiffusionTrainer(BaseTrainer):
    def setup_dataloader(self):
        # sub_paths: list[Path] = []
        # if self.multi_aspect_training:
        #     for sub_path in Path(self.dataset_name_or_path).glob("*_*"):
        #         if not sub_path.is_dir():
        #             continue

        #         sub_paths.append(sub_path.resolve())
        # else:
        #     sub_paths.append(Path(self.dataset_name_or_path).resolve())

        # streams: list[Stream] = []
        # for sub_path in sub_paths:
        #     match = re.match("^([0-9]+)_([0-9]+)_([0-9]+)$", sub_path.stem)
        #     if match:
        #         num_samples = int(match.group(3))
        #         target_size = (int(match.group(1)), int(match.group(2)))

        #         # TODO: fix me
        #         if num_samples < 2048:
        #             continue
        #     else:
        #         num_samples = 1
        #         target_size = None

        #     if self.resolution is not None:
        #         if isinstance(self.resolution, int):
        #             target_size = (self.resolution, self.resolution)
        #         else:
        #             target_size = self.resolution

        #     Stream(remote=sub_path, proportion=)

        ds = MMStreamingDataset(
            dataset_path=self.dataset_name_or_path,
            model_name_or_path=self.model_name_or_path,
            model_revision=self.model_revision,
            num_tokenizers=self.num_text_encoders,
            resolution=self.resolution,
            proportion_empty_prompts=self.proportion_empty_prompts,
            center_crop=self.center_crop,
            random_flip=self.random_flip,
            shuffle_seed=self.seed,
        )

        return DataLoader(
            ds,
            batch_size=self.train_batch_size,
            num_workers=self.num_dataset_workers,
            pin_memory=True,
        )

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

        model_input = compute_vae_encodings(batch["image"], vae)
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
