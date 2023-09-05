import os
import importlib
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from jsonargparse import ArgumentParser, ActionConfigFile

from .configs import (
    DataConfig, ModelConfig, OptimizerConfig, TrainerConfig, LoggingConfig, HubConfig
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

        self.config = self.parser.parse_args(args)

        # Instantiate
        self.config_instantiated = self.parser.instantiate_classes(self.config)

        subcommand = self.config["subcommand"]

        self.trainer = self.config_instantiated[subcommand]["trainer"]

        config = self.config_instantiated[subcommand]
        # Run subcomman
        getattr(self.trainer, subcommand)(
            data_config=config["data"],
            model_config=config["model"],
            optimizer_config=config["optimizer"],
        )

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

        # parser.add_dataclass_arguments(
        #     TrainerConfig,
        #     nested_key="trainer",
        # )

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
