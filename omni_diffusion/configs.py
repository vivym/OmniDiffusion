from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset_name: str

    image_column: str = "image"

    caption_column: str = "text"

    num_workers: int = 0

    resolution: int = 512

    center_crop: bool = False

    random_flip: bool = False


@dataclass
class ModelConfig:
    model_name_or_path: str

    revision: str | None = None


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4

    auto_scale_lr: bool = False

    use_8bit_adam: bool = False

    weight_decay: float = 1e-2

    adam_beta: tuple[float, float] = (0.9, 0.999)

    adam_epsilon: float = 1e-8

    # The scheduler type to use.
    # Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    lr_scheduler: str | None = None

    lr_warmup_steps: int = 500


@dataclass
class TrainerConfig:
    project_name: str = "omni-diffusion"

    output_dir: str = "./outputs"

    max_epochs: int = 10

    max_steps: int | None = None

    num_validation_samples: int = 4

    validation_every_n_steps: int = 200

    validation_prompt: str | None = None

    train_batch_size: int = 16

    gradient_accumulation_steps: int = 1

    gradient_checkpointing: bool = False

    checkpointing_steps: int = 500

    max_checkpoints: int | None = None

    resume_from_checkpoint: str | None = None

    # SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0.
    # More details here: https://arxiv.org/abs/2303.09556.
    snr_gamma: float | None = None

    # When using SNR gamma with rescaled betas for zero terminal SNR, a divide-by-zero error can cause NaN
    # condition when computing the SNR with a sigma value of zero. This parameter overrides the check,
    # allowing the use of SNR gamma with a terminal SNR model. Use with caution, and closely monitor results.
    force_snr_gamma: bool = False

    # Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training.
    allow_tf32: bool = False

    max_grad_norm: float = 1.0

    use_ema: bool = False

    # The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`.
    # If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.
    prediction_type: str | None = None

    mixed_precision: str | None = None      # no, fp16, bf16

    use_xformers: bool = False

    # The scale of noise offset.
    noise_offset: float = 0.0

    # Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).
    proportion_empty_prompts: float = 0.0

    seed: bool | int = True

    report_to: str = "tensorboard"


@dataclass
class LoggingConfig:
    logging_dir: str = "logs"


@dataclass
class HubConfig:
    push_to_hub: bool = False

    model_id: str | None = None

    commit_message: str | None = None

    private: bool = False
