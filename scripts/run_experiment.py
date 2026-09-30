#!/usr/bin/env python3
"""Run YAML-defined experiment commands without embedding a model in Bash."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--n-jobs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--train-n-jobs", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def format_argv(argv: Sequence[object], values: Mapping[str, str]) -> list[str]:
    try:
        return [str(argument).format_map(values) for argument in argv]
    except KeyError as error:
        raise ValueError(f"unknown command placeholder: {error.args[0]}") from error


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, Mapping):
        raise ValueError(f"YAML config must contain a mapping: {config_path}")
    commands = config.get("commands")
    if not isinstance(commands, Sequence) or isinstance(commands, (str, bytes)):
        raise ValueError("YAML config requires a non-empty 'commands' list")
    if not commands:
        raise ValueError("YAML config requires at least one command")

    values = {
        "config": str(config_path),
        "run_dir": str(args.run_dir.expanduser().resolve()),
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "device": args.device,
        "n_jobs": str(args.n_jobs),
        "batch_size": str(args.batch_size),
        "train_n_jobs": str(args.train_n_jobs),
        "nproc_per_node": str(args.nproc_per_node),
        "python": sys.executable,
        "torchrun": shutil.which("torchrun") or "torchrun",
    }
    for index, command in enumerate(commands, start=1):
        if not isinstance(command, Mapping):
            raise ValueError(f"commands[{index}] must be a mapping")
        name = command.get("name", f"command-{index}")
        argv = command.get("argv")
        if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)) or not argv:
            raise ValueError(f"commands[{index}].argv must be a non-empty list")
        rendered_argv = format_argv(argv, values)
        print(f"===== {name} =====", flush=True)
        print(f"$ {shlex.join(rendered_argv)}", flush=True)
        if not args.dry_run:
            subprocess.run(rendered_argv, check=True)


if __name__ == "__main__":
    main()
