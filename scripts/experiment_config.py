"""YAML defaults shared by Transformer training and evaluation entry points."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


def load_config_defaults(
    argv: list[str] | None,
    *,
    sections: Iterable[str],
    valid_destinations: set[str],
) -> dict[str, Any]:
    """Read selected YAML sections as argparse defaults.

    Command-line arguments are parsed afterwards and therefore intentionally
    take precedence over the YAML file.
    """
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    config_args, _ = bootstrap.parse_known_args(argv)
    if config_args.config is None:
        return {}

    config_path = config_args.config.expanduser()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, Mapping):
        raise ValueError(f"YAML config must contain a mapping: {config_path}")

    defaults: dict[str, Any] = {}
    for section in sections:
        values = config.get(section, {})
        if not isinstance(values, Mapping):
            raise ValueError(f"YAML section '{section}' must be a mapping: {config_path}")
        for key, value in values.items():
            destination = str(key).replace("-", "_")
            if destination not in valid_destinations:
                raise ValueError(
                    f"unsupported key '{section}.{key}' in {config_path}; "
                    "use a command-line option name"
                )
            defaults[destination] = value
    return defaults
