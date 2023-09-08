import math
import os
import io
import itertools
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from accelerate.logging import get_logger
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TrF
from tqdm import tqdm
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel

from omni_diffusion.configs import (
    DataConfig, ModelConfig, OptimizerConfig
)
from .base_trainer import BaseTrainer

logger = get_logger(__name__, log_level="INFO")


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


def compute_vae_encodings(batch, vae: AutoencoderKL) -> torch.Tensor:
    pixel_values = batch.pop("pixel_values")

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    return model_input * vae.config.scaling_factor


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


def encode_prompts(
    batch,
    text_encoder_one: PreTrainedModel,
    text_encoder_two: PreTrainedModel,
    proportion_empty_prompts: float = 0.0,
    is_train: bool = True,
):
    # TODO: proportion_empty_prompts

    prompt_embeds_list = []

    input_ids_list = [batch["input_ids_one"], batch["input_ids_two"]]
    text_encoders = [text_encoder_one, text_encoder_two]

    with torch.no_grad():
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


class SDXLTrainer(BaseTrainer):
    def fit(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
    ):
        accelerator = self.setup_accelerator()

        if accelerator.is_main_process:
            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)

            if self.push_to_hub:
                repo_id = create_repo(
                    repo_id=self.hub_model_id or Path(self.output_dir).name,
                    exist_ok=True,
                ).repo_id

        accelerator.wait_for_everyone()

        # Load scheduler and models.
        noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.revision,
            subfolder="scheduler",
        )
        # Check for terminal SNR in combination with SNR Gamma
        if (
            self.snr_gamma is not None
            and not self.force_snr_gamma
            and (
                hasattr(noise_scheduler.config, "rescale_betas_zero_snr") and noise_scheduler.config.rescale_betas_zero_snr
            )
        ):
            raise ValueError(
                f"The selected noise scheduler for the model `{model_config.model_name_or_path}` uses rescaled betas for zero SNR.\n"
                "When this configuration is present, the parameter `snr_gamma` may not be used without parameter `force_snr_gamma`.\n"
                "This is due to a mathematical incompatibility between our current SNR gamma implementation, and a sigma value of zero."
            )

        # Load the tokenizers.
        tokenizer_one = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.revision,
            subfolder="tokenizer",
            use_fast=True, # TODO: support fast tokenizers
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.revision,
            subfolder="tokenizer_2",
            use_fast=True, # TODO: support fast tokenizers
        )

        # import text encoder classes
        text_encode_cls_one = import_model_class_from_model_name_or_path(
            model_config.model_name_or_path,
            revision=model_config.revision,
            subfolder="text_encoder",
        )

        text_encode_cls_two = import_model_class_from_model_name_or_path(
            model_config.model_name_or_path,
            revision=model_config.revision,
            subfolder="text_encoder_2",
        )

        text_encoder_one = text_encode_cls_one.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.revision,
            subfolder="text_encoder",
        )
        text_encoder_two = text_encode_cls_two.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.revision,
            subfolder="text_encoder_2",
        )

        vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.revision,
            subfolder="vae",
        )

        unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.revision,
            subfolder="unet",
        )

        # Freeze vae and text encoders.
        vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        if self.use_lora:
            # We only train the additional adapter LoRA layers.
            unet.requires_grad_(False)

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
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)

        if self.use_lora:
            unet.to(accelerator.device, dtype=weight_dtype)

        if self.use_ema:
            assert not self.use_lora, "EMA is not supported with LoRA"

            ema_unet_base = UNet2DConditionModel.from_pretrained(
                model_config.model_name_or_path,
                revision=model_config.revision,
                subfolder="unet",
            )
            ema_unet = EMAModel(
                ema_unet_base,
                model_cls=UNet2DConditionModel,
                model_config = ema_unet_base.config,
            )

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
                text_lora_parameters_two = LoraLoaderMixin._modify_text_encoder(
                    text_encoder_two, dtype=torch.float32, rank=self.lora_rank
                )

            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                # there are only two options here. Either are just the unet attn processor layers
                # or there are the unet and text encoder atten layers
                unet_lora_layers_to_save = None
                text_encoder_one_lora_layers_to_save = None
                text_encoder_two_lora_layers_to_save = None

                for model in models:
                    if isinstance(model, type(accelerator.unwrap_model(unet))):
                        unet_lora_layers_to_save = unet_attn_processors_state_dict(model)
                    elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                        text_encoder_one_lora_layers_to_save = text_encoder_lora_state_dict(model)
                    elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                        text_encoder_two_lora_layers_to_save = text_encoder_lora_state_dict(model)
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
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

                while len(models) > 0:
                    model = models.pop()

                    if isinstance(model, type(accelerator.unwrap_model(unet))):
                        unet_ = model
                    elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                        text_encoder_one_ = model
                    elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                        text_encoder_two_ = model
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
                LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)

                text_encoder_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder." in k}
                LoraLoaderMixin.load_lora_into_text_encoder(
                    text_encoder_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_one_
                )

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

        if self.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        dataset_dict = load_dataset(
            data_config.dataset_name,
            # TODO: cache dir
        )

        column_names = dataset_dict["train"].column_names

        if data_config.image_column not in column_names:
            raise ValueError(
                f"Image column name `{data_config.image_column}` should be one of {', '.join(column_names)}."
            )

        if data_config.caption_column not in column_names:
            raise ValueError(
                f"Caption column name `{data_config.image_column}` should be one of {', '.join(column_names)}."
            )

        train_resize = T.Resize(data_config.resolution, interpolation=T.InterpolationMode.BILINEAR)
        train_crop = T.CenterCrop(data_config.resolution) if data_config.center_crop else T.RandomCrop(data_config.resolution)
        train_flip = T.RandomHorizontalFlip(p=1.0)
        train_normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.5],     # TODO: check if this is correct
                std=[0.5],
                # mean=[0.48145466, 0.4578275, 0.40821073],
                # std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

        def tokenize_captions(samples, is_train: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
            captions = []

            for caption in samples[data_config.caption_column]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    captions.append(random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(f"Unknown caption type: {type(caption)}")

            input_ids_one = tokenizer_one(
                captions,
                padding="max_length",
                truncation=True,
                max_length=tokenizer_one.model_max_length,
                return_tensors="pt",
            ).input_ids

            input_ids_two = tokenizer_two(
                captions,
                padding="max_length",
                truncation=True,
                max_length=tokenizer_two.model_max_length,
                return_tensors="pt",
            ).input_ids

            return input_ids_one, input_ids_two

        def preprocess_images(samples, is_train: bool = True) -> torch.Tensor:
            images = [
                Image.open(io.BytesIO(image_data))
                for image_data in samples[data_config.image_column]
            ]

            original_sizes = []
            all_images = []
            crop_top_lefts = []

            for image in images:
                original_sizes.append((image.height, image.width))
                image = train_resize(image)

                if data_config.center_crop or is_train:
                    y1 = max(0, int(round(image.height - data_config.resolution) / 2.0))
                    x1 = max(0, int(round(image.width - data_config.resolution) / 2.0))
                    image = train_crop(image)
                else:
                    y1, x1, h, w = train_crop.get_params(image, (data_config.resolution, data_config.resolution))
                    image = TrF.crop(image, y1, x1, h, w)

                if data_config.random_flip and is_train and random.random() < 0.5:
                    x1 = image.width - x1
                    image = train_flip(image)

                crop_top_lefts.append((y1, x1))
                image = train_normalize(image)
                all_images.append(image)

            return all_images, original_sizes, crop_top_lefts

        def preprocess_train(samples):
            input_ids_one, input_ids_two = tokenize_captions(samples)
            samples["input_ids_one"] = input_ids_one
            samples["input_ids_two"] = input_ids_two

            images, original_sizes, crop_top_lefts = preprocess_images(samples)
            samples["pixel_values"] = images
            samples["original_sizes"] = original_sizes
            samples["crop_top_lefts"] = crop_top_lefts

            return samples

        with accelerator.main_process_first():
            train_dataset = dataset_dict["train"].with_transform(preprocess_train)

        def collate_fn(samples):
            pixel_values = torch.stack([sample["pixel_values"] for sample in samples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids_one = torch.stack([sample["input_ids_one"] for sample in samples])
            input_ids_two = torch.stack([sample["input_ids_two"] for sample in samples])
            original_sizes = [sample["original_sizes"] for sample in samples]
            crop_top_lefts = [sample["crop_top_lefts"] for sample in samples]
            return {
                "pixel_values": pixel_values,
                "input_ids_one": input_ids_one,
                "input_ids_two": input_ids_two,
                "original_sizes": original_sizes,
                "crop_top_lefts": crop_top_lefts,
            }

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            num_workers=data_config.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
        )

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        if self.max_steps is None:
            self.max_steps = self.max_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True
        else:
            overrode_max_train_steps = False

        if optimizer_config.auto_scale_lr:
            optimizer_config.learning_rate = (
                optimizer_config.learning_rate * self.gradient_accumulation_steps * self.train_batch_size * accelerator.num_processes
            )

        if self.use_lora:
            params_to_optimize = unet_lora_parameters

            if self.train_text_encoder:
                params_to_optimize = itertools.chain(
                    params_to_optimize, text_lora_parameters_one, text_lora_parameters_two
                )
        else:
            params_to_optimize = unet.parameters()

        optimizer, lr_scheduler = self.setup_optimizer(
            params_to_optimize,
            optimizer_config=optimizer_config,
        )

        if self.train_text_encoder:
            train_dataloader, unet, text_encoder_one, text_encoder_two, optimizer, lr_scheduler = accelerator.prepare(
                train_dataloader, unet, text_encoder_one, text_encoder_two, optimizer, lr_scheduler
            )
        else:
            train_dataloader, unet, optimizer, lr_scheduler = accelerator.prepare(
                train_dataloader, unet, optimizer, lr_scheduler
            )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.max_steps = self.max_epochs * num_update_steps_per_epoch
        self.max_epochs = math.ceil(self.max_steps / num_update_steps_per_epoch)

        if accelerator.is_main_process:
            accelerator.init_trackers(
                self.project_name,
                config={
                    "data": data_config,
                    "model": model_config,
                    "optimizer": optimizer_config,
                    "trainer": self.config,
                },
            )

        total_batch_size = self.train_batch_size * accelerator.num_processes * self.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {self.max_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_steps}")
        global_step = 0
        first_epoch = 0

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

                resume_global_step = global_step * self.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (num_update_steps_per_epoch * self.gradient_accumulation_steps)

        progress_bar = tqdm(
            range(global_step, self.max_steps),
            desc="Steps",
            disable=not accelerator.is_local_main_process
        )

        for epoch in range(first_epoch, self.max_epochs):
            unet.train()
            if self.train_text_encoder:
                text_encoder_one.train()
                text_encoder_two.train()

            train_loss = 0.0

            for step, batch in enumerate(train_dataloader):
                if self.resume_from_checkpoint is not None and epoch == first_epoch and step < resume_step:
                    if step % self.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                prompt_embeds, pooled_prompt_embeds = encode_prompts(
                    batch,
                    text_encoder_one=text_encoder_one,
                    text_encoder_two=text_encoder_two,
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
                    device=model_input.device,
                )
                timesteps = timesteps.long()

                # time ids
                def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (data_config.resolution, data_config.resolution)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids

                with accelerator.accumulate(unet):
                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                    add_time_ids = torch.cat(
                        [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                    )

                    # Predict the noise residual
                    unet_added_conditions = {"time_ids": add_time_ids}
                    unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})

                    model_pred = unet(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds,
                        added_cond_kwargs=unet_added_conditions
                    ).sample

                    # Get the target for loss depending on the prediction type
                    if self.prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        noise_scheduler.register_to_config(prediction_type=self.prediction_type)
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
                        mse_loss_weights = (
                            torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                        )
                        # We first calculate the original loss. Then we mean over the non-batch dimensions and
                        # rebalance the sample-wise losses with their respective loss weights.
                        # Finally, we take the mean of the rebalanced loss.
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(self.train_batch_size)).mean()
                    train_loss += avg_loss.item() / self.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = unet.parameters()

                        if self.use_lora:
                            params_to_clip = unet_lora_parameters

                            if self.train_text_encoder:
                                params_to_clip = itertools.chain(
                                    params_to_clip, text_lora_parameters_one, text_lora_parameters_two
                                )
                        else:
                            params_to_clip = unet.parameters()

                        accelerator.clip_grad_norm_(params_to_clip, self.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if accelerator.is_main_process:
                        if global_step % self.checkpointing_every_n_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if self.max_checkpoints is not None:
                                checkpoints = os.listdir(self.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `max_checkpoints - 1` checkpoints
                                if len(checkpoints) >= self.max_checkpoints:
                                    num_to_remove = len(checkpoints) - self.max_checkpoints + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(self.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= self.max_steps:
                    break

            if accelerator.is_main_process:
                # TODO: validation_every_n_steps
                if self.validation_prompt is not None and epoch % self.validation_every_n_steps == 0:
                    logger.info(
                        f"Running validation... \n Generating {self.num_validation_samples} images with prompt:"
                        f" {self.validation_prompt}."
                    )
                    if self.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())

                    # create pipeline
                    # vae = AutoencoderKL.from_pretrained(
                    #     model_config.model_name_or_path,
                    #     revision=model_config.revision,
                    #     subfolder="vae",
                    # )
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        model_config.model_name_or_path,
                        vae=vae,
                        text_encoder=accelerator.unwrap_model(text_encoder_one),
                        text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                        unet=accelerator.unwrap_model(unet),
                        revision=model_config.revision,
                        torch_dtype=weight_dtype,
                    )
                    if self.prediction_type is not None:
                        scheduler_args = {"prediction_type": self.prediction_type}
                        pipeline.scheduler = pipeline.scheduler.from_config(
                            pipeline.scheduler.config,
                            **scheduler_args,
                        )

                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    # run inference
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
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            import wandb

                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {self.validation_prompt}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )

                    del pipeline
                    torch.cuda.empty_cache()

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unet = accelerator.unwrap_model(unet)
            if self.use_ema:
                ema_unet.copy_to(unet.parameters())

            if self.train_text_encoder:
                text_encoder_one = accelerator.unwrap_model(text_encoder_one)
                text_encoder_lora_layers = text_encoder_lora_state_dict(text_encoder_one)
                text_encoder_two = accelerator.unwrap_model(text_encoder_two)
                text_encoder_2_lora_layers = text_encoder_lora_state_dict(text_encoder_two)
            else:
                text_encoder_lora_layers = None
                text_encoder_2_lora_layers = None

            if self.use_lora:
                unet_lora_layers = unet_attn_processors_state_dict(unet)

                StableDiffusionXLPipeline.save_lora_weights(
                    save_directory=self.output_dir,
                    unet_lora_layers=unet_lora_layers,
                    text_encoder_lora_layers=text_encoder_lora_layers,
                    text_encoder_2_lora_layers=text_encoder_2_lora_layers,
                )

                del unet
                del text_encoder_one
                del text_encoder_two
                del text_encoder_lora_layers
                del text_encoder_2_lora_layers
                torch.cuda.empty_cache()

                # Final inference
                # Load previous pipeline
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_config.model_name_or_path, vae=vae, revision=model_config.revision, torch_dtype=weight_dtype
                )
                pipeline = pipeline.to(accelerator.device)

                # load attention processors
                pipeline.load_lora_weights(self.output_dir)
            else:
                # Serialize pipeline.
                vae = AutoencoderKL.from_pretrained(
                    model_config.model_name_or_path,
                    revision=model_config.revision,
                    subfolder="vae",
                    torch_dtype=weight_dtype,
                )
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_config.model_name_or_path,
                    unet=unet,
                    vae=vae,
                    revision=model_config.revision,
                    torch_dtype=weight_dtype,
                )
                if self.prediction_type is not None:
                    scheduler_args = {"prediction_type": self.prediction_type}
                    pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
                pipeline.save_pretrained(self.output_dir)

            # run inference
            images = []
            if self.validation_prompt and self.num_validation_samples > 0:
                pipeline = pipeline.to(accelerator.device)
                generator = torch.Generator(device=accelerator.device).manual_seed(self.seed) if self.seed else None
                with torch.cuda.amp.autocast():
                    images = [
                        pipeline(self.validation_prompt, generator=generator).images[0]
                        for _ in range(self.num_validation_samples)
                    ]

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "test": [
                                    wandb.Image(image, caption=f"{i}: {self.validation_prompt}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

            if self.push_to_hub:
                # save_model_card(
                #     repo_id=repo_id,
                #     images=images,
                #     validation_prompt=self.validation_prompt,
                #     base_model=model_config.model_name_or_path,
                #     dataset_name=data_config.dataset_name,
                #     repo_folder=self.output_dir,
                #     vae_path=model_config.model_name_or_path,
                # )
                upload_folder(
                    repo_id=repo_id,
                    folder_path=self.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        accelerator.end_training()

    def validate(self):
        ...

    def test(self):
        ...

    def predict(self):
        ...
