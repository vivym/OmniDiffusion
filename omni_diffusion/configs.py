from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset_name_or_path: str | list[str]

    revision: str | None = None

    image_column: str = "image"

    prompt_column: str | list[str] = "prompt"

    num_workers: int = 16

    resolution: int | tuple[int, int] | None = None

    center_crop: bool = False

    random_flip: bool = False

    # Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).
    proportion_empty_prompts: float = 0.0

    multi_aspect_training: bool = False

    local_shuffle_buffer_size: int = 128

    prefetch_batches: int = 4


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
    lr_scheduler: str = "constant"

    lr_warmup_steps: int = 0


@dataclass
class LoggingConfig:
    logging_dir: str = "logs"

    report_to: str = "tensorboard"


@dataclass
class HubConfig:
    push_to_hub: bool = False

    model_id: str | None = None

    commit_message: str | None = None

    private: bool = False
