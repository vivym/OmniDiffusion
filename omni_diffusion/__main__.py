import os
import importlib
from typing import Any, TYPE_CHECKING

import ray

from jsonargparse import ArgumentParser, ActionConfigFile
from ray.train import CheckpointConfig, RunConfig, ScalingConfig, DataConfig as RayDataConfig
from ray.train.torch import TorchTrainer

from .configs import (
    DataConfig, ModelConfig, OptimizerConfig, LoggingConfig, HubConfig
)
from .trainers import BaseTrainer

if TYPE_CHECKING:
    from jsonargparse._actions import _ActionSubCommands


def instantiate_class(args: Any | tuple[Any, ...], init: dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})

    if not isinstance(args, tuple):
        args = (args,)

    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = importlib.import_module(class_module)
    cls = getattr(module, class_name)

    return cls(*args, **kwargs)


class OmniArgumentParser(ArgumentParser):
    def __init__(
        self,
        *args: Any,
        description: str = "OmniDiffusion CLI",
        env_prefix: str = "OMNID",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            description=description,
            env_prefix=env_prefix,
            **kwargs,
        )


class OmniCLI:
    """OmniAI CLI"""

    def __init__(self, args: list[str] | None = None) -> None:
        self.parser = self.init_parser()

        subcommands = self.parser.add_subcommands(required=True)
        self.add_subcommand(
            "fit",
            help="Full training pipeline",
            subcommands=subcommands,
        )
        self.add_subcommand(
            "validate",
            help="Validation pipeline",
            subcommands=subcommands,
        )
        self.add_subcommand(
            "test",
            help="Test pipeline",
            subcommands=subcommands,
        )
        self.add_subcommand(
            "predict",
            help="Prediction pipeline",
            subcommands=subcommands,
        )

        config = self.parser.parse_args(args)

        # Instantiate
        subcommand = config["subcommand"]
        self.config = self.parser.instantiate_classes(config)[subcommand]

        trainer: BaseTrainer = self.config["trainer"]

        os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = trainer.output_dir

        import json
        ray.init(
            runtime_env={
                "env_vars": {
                    "RAY_AIR_LOCAL_CACHE_DIR": os.environ["RAY_AIR_LOCAL_CACHE_DIR"],
                },
                "working_dir": ".",
            },
            # object_store_memory=700 * 1024 * 1024 * 1024,
            _system_config={
                "max_io_workers": 4,
                "object_spilling_config": json.dumps({
                    "type": "filesystem",
                    "params": {
                        "directory_path": [
                            "/home/mingyang/tmp",
                            "/home2/mingyang/tmp",
                        ],
                    },
                }),
            },
        )

        trainer.prepare_configs(
            data_config=self.config["data"],
            model_config=self.config["model"],
            optimizer_config=self.config["optimizer"],
            logging_config=self.config["logging"],
            hub_config=self.config["hub"],
        )

        datasets = trainer.prepare_data()
        dataset_names = list(datasets.keys())

        trainer.output_dir = os.path.realpath(trainer.output_dir)

        os.makedirs(trainer.output_dir, exist_ok=True)
        artifact_dir = os.path.join(trainer.output_dir, "artifact")
        os.makedirs(artifact_dir, exist_ok=True)

        if trainer.push_to_hub:
            from huggingface_hub import create_repo

            trainer.repo_id = create_repo(
                repo_id=trainer.hub_model_id or os.path.basename(trainer.output_dir),
                private=trainer.private_hub,
                exist_ok=True,
            ).repo_id

        trainer = TorchTrainer(
            getattr(trainer, subcommand),
            run_config=RunConfig(
                name=trainer.project_name,
                storage_path=artifact_dir,
                checkpoint_config=CheckpointConfig(
                    num_to_keep=trainer.max_checkpoints,
                ),
                log_to_file=False,
            ),
            scaling_config=ScalingConfig(
                num_workers=trainer.num_devices,
                use_gpu=True,
                trainer_resources={"CPU": 4},
                resources_per_worker={"CPU": 4, "GPU": 1},
            ),
            datasets=datasets,
            dataset_config=RayDataConfig(datasets_to_split=dataset_names),
        )
        trainer.fit()

    def add_subcommand(
        self,
        name: str,
        help: str,
        subcommands: "_ActionSubCommands",
    ) -> None:
        parser = self.init_parser()
        parser.add_argument(
            "--seed",
            type=bool | int,
            default=True,
            help=(
                "Random seed. "
                "If True, a random seed will be generated. "
                "If False, no random seed will be used. "
                "If an integer, that integer will be used as the random seed."
            ),
        )

        parser.add_dataclass_arguments(
            DataConfig,
            nested_key="data",
        )

        parser.add_dataclass_arguments(
            ModelConfig,
            nested_key="model",
        )

        parser.add_dataclass_arguments(
            OptimizerConfig,
            nested_key="optimizer",
        )

        parser.add_subclass_arguments(
            BaseTrainer,
            nested_key="trainer",
            required=True,
            fail_untyped=False,
        )

        parser.add_dataclass_arguments(
            LoggingConfig,
            nested_key="logging",
            default=LoggingConfig(),
        )

        parser.add_dataclass_arguments(
            HubConfig,
            nested_key="hub",
            default=HubConfig(),
        )

        subcommands.add_subcommand(name, parser, help=help)

    def init_parser(self) -> OmniArgumentParser:
        parser = OmniArgumentParser()
        parser.add_argument(
            "-c",
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format.",
        )
        return parser


if __name__ == "__main__":
    OmniCLI()
