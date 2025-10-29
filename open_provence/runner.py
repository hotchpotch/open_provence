"""
Training runner module for OpenProvence models.

This module provides the main training functionality for OpenProvence models,
supporting various configurations through command line arguments and config files.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from transformers.hf_argparser import HfArgumentParser

from .trainer import (
    DataArguments,
    ModelArguments,
    PruningTrainingArguments,
    parse_config_file,
    run_eval_datasets_for_model,
    train,
)

logger = logging.getLogger(__name__)


def _warn_if_flash_attn_missing() -> None:
    try:
        import flash_attn  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ModuleNotFoundError:
        logger.warning(
            "flash-attn is not available. Adding an implementation such as "
            "https://github.com/Dao-AILab/flash-attention can significantly speed up training "
            "on some CUDA environments.",
        )


def run_training(
    config_file: str | None = None,
    **kwargs,
) -> str:
    """
    Run training with the given configuration.

    Args:
        config_file: Path to YAML/JSON configuration file
        **kwargs: Additional arguments to override config

    Returns:
        Path to the final trained model
    """
    parser = HfArgumentParser((ModelArguments, DataArguments, PruningTrainingArguments))  # type: ignore[arg-type]

    if config_file:
        # Parse config file first
        print(f"Loading configuration from: {config_file}")
        if config_file.endswith(".json"):
            model_args, data_args, training_args = parser.parse_json_file(json_file=config_file)
        else:
            model_args, data_args, training_args = parse_config_file(config_file)

        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(model_args, key):
                setattr(model_args, key, value)
            elif hasattr(data_args, key):
                setattr(data_args, key, value)
            elif hasattr(training_args, key):
                setattr(training_args, key, value)
    else:
        # Create args from kwargs
        model_args = ModelArguments(
            **{k: v for k, v in kwargs.items() if hasattr(ModelArguments, k)}
        )
        data_args = DataArguments(**{k: v for k, v in kwargs.items() if hasattr(DataArguments, k)})
        training_args = PruningTrainingArguments(
            **{k: v for k, v in kwargs.items() if hasattr(PruningTrainingArguments, k)}
        )

    # Create timestamp for unique naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set output_dir if not specified
    if not training_args.output_dir or training_args.output_dir == "trainer_output":
        # Generate default output_dir with timestamp
        if config_file:
            # Use config file name as base
            config_base = Path(config_file).stem
            output_dir = f"./output/{config_base}_{timestamp}"
        else:
            # Fallback to old naming scheme
            model_name = Path(model_args.model_name_or_path).name
            if data_args.datasets:
                # For multi-dataset configs, use a generic name
                output_dir = f"./output/{model_name}_multi-dataset_{timestamp}"
            else:
                output_dir = f"./output/{model_name}_{data_args.subset}_{timestamp}"

        training_args.output_dir = output_dir
        print(f"\n{'=' * 80}")
        print("No output_dir specified. Auto-generated output directory:")
        print(f"  {output_dir}")
        print(f"{'=' * 80}\n")

    # Create run name based on configuration with timestamp
    if config_file:
        # Use config file name as base for run name
        config_base = Path(config_file).stem
        run_name = f"{config_base}-{timestamp}"
    else:
        # Fallback to old naming scheme
        model_name = Path(model_args.model_name_or_path).name
        if data_args.datasets:
            run_name = f"{model_name}-multi-dataset-{timestamp}"
        else:
            run_name = f"{model_name}-{data_args.subset}-{timestamp}"

    # Call train function from trainer module
    final_model_path = train(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        run_name=run_name,
        timestamp=timestamp,
    )

    print(f"Training completed. Final model saved to: {final_model_path}")

    return final_model_path


def main():
    """CLI entry point for training."""
    _warn_if_flash_attn_missing()

    # Parse arguments - either from command line or config files
    parser = HfArgumentParser((ModelArguments, DataArguments, PruningTrainingArguments))  # type: ignore[arg-type]

    # Replace sys.argv temporarily (may be modified by parsing logic below)
    original_argv = sys.argv
    working_argv = sys.argv[:]

    eval_datasets_model_path: str | None = None
    filtered_argv = [working_argv[0]]
    i = 1
    while i < len(working_argv):
        arg = working_argv[i]
        if arg in {"--eval-datasets-model", "--only-eval-datasets-model"}:
            if i + 1 >= len(working_argv):
                raise ValueError(f"{arg} requires a model path argument")
            eval_datasets_model_path = working_argv[i + 1]
            i += 2
        else:
            filtered_argv.append(arg)
            i += 1

    sys.argv = filtered_argv

    try:
        # Check if first argument is a config file
        config_file_arg: str | None = None
        model_args = None
        data_args = None
        training_args = None
        if len(sys.argv) >= 2 and sys.argv[1].endswith((".yaml", ".yml", ".json")):
            config_file_arg = sys.argv[1]
            # Remove config file from argv to parse remaining args
            remaining_args = sys.argv[2:]
        else:
            remaining_args = sys.argv[1:]

        if config_file_arg:
            # Parse config file first
            print(f"Loading configuration from: {config_file_arg}")
            if config_file_arg.endswith(".json"):
                model_args, data_args, training_args = parser.parse_json_file(
                    json_file=config_file_arg
                )
            else:
                model_args, data_args, training_args = parse_config_file(config_file_arg)

            # Parse remaining command line arguments to override config
            if remaining_args:
                # Create temporary argv for remaining args
                temp_argv = [sys.argv[0]] + remaining_args
                original_argv2 = sys.argv
                sys.argv = temp_argv
                try:
                    override_model_args, override_data_args, override_training_args = (
                        parser.parse_args_into_dataclasses()
                    )

                    # Override config values with command line values (only non-default values)
                    overrides = []

                    for field_name in model_args.__dataclass_fields__:
                        override_value = getattr(override_model_args, field_name)
                        default_value = model_args.__dataclass_fields__[field_name].default
                        # Handle special case for required fields that don't have real defaults
                        if field_name == "model_name_or_path" and hasattr(
                            override_model_args, field_name
                        ):
                            # Always override model_name_or_path if provided
                            old_value = getattr(model_args, field_name)
                            if old_value != override_value:
                                setattr(model_args, field_name, override_value)
                                overrides.append(
                                    f"model_args.{field_name}: {old_value} → {override_value}"
                                )
                        elif override_value != default_value:
                            old_value = getattr(model_args, field_name)
                            setattr(model_args, field_name, override_value)
                            overrides.append(
                                f"model_args.{field_name}: {old_value} → {override_value}"
                            )

                    for field_name in data_args.__dataclass_fields__:
                        override_value = getattr(override_data_args, field_name)
                        default_value = data_args.__dataclass_fields__[field_name].default
                        if override_value != default_value:
                            old_value = getattr(data_args, field_name)
                            setattr(data_args, field_name, override_value)
                            overrides.append(
                                f"data_args.{field_name}: {old_value} → {override_value}"
                            )

                    for field_name in training_args.__dataclass_fields__:
                        override_value = getattr(override_training_args, field_name)
                        default_value = training_args.__dataclass_fields__[field_name].default
                        if override_value != default_value:
                            old_value = getattr(training_args, field_name)
                            setattr(training_args, field_name, override_value)
                            overrides.append(
                                f"training_args.{field_name}: {old_value} → {override_value}"
                            )

                    # Log overrides
                    if overrides:
                        print("Command line overrides:")
                        for override in overrides:
                            print(f"  {override}")
                    else:
                        print("No command line overrides applied.")

                finally:
                    sys.argv = original_argv2
            else:
                print("Using configuration file settings.")
        else:
            # No config file, parse command line only
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    finally:
        # Restore original sys.argv
        sys.argv = original_argv

    if training_args is None:
        raise RuntimeError("Failed to parse training arguments.")

    if eval_datasets_model_path:
        eval_settings = getattr(training_args, "eval_datasets", None)
        if not eval_settings:
            print("No eval_datasets configuration found; nothing to evaluate.")
            return
        run_eval_datasets_for_model(eval_datasets_model_path, eval_settings)
        return

    if model_args is None or data_args is None:
        raise RuntimeError("Failed to parse model/data arguments.")

    # Create timestamp for unique naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set output_dir if not specified or using default
    if not training_args.output_dir or training_args.output_dir == "trainer_output":
        # Generate default output_dir with timestamp
        if config_file_arg:
            # Use config file name as base
            config_base = Path(config_file_arg).stem
            output_dir = f"./output/{config_base}_{timestamp}"
        else:
            # Fallback to old naming scheme
            model_name = Path(model_args.model_name_or_path).name
            if data_args.datasets:
                # For multi-dataset configs, use a generic name
                output_dir = f"./output/{model_name}_multi-dataset_{timestamp}"
            else:
                output_dir = f"./output/{model_name}_{data_args.subset}_{timestamp}"

        training_args.output_dir = output_dir
        print(f"\n{'=' * 80}")
        print("No output_dir specified. Auto-generated output directory:")
        print(f"  {output_dir}")
        print(f"{'=' * 80}\n")

    # Create run name based on configuration with timestamp
    if config_file_arg:
        # Use config file name as base for run name
        config_base = Path(config_file_arg).stem
        run_name = f"{config_base}-{timestamp}"
    else:
        # Fallback to old naming scheme
        model_name = Path(model_args.model_name_or_path).name
        if data_args.datasets:
            run_name = f"{model_name}-multi-dataset-{timestamp}"
        else:
            run_name = f"{model_name}-{data_args.subset}-{timestamp}"

    # Call train function from trainer module
    train(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        run_name=run_name,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
