from pathlib import Path

import fire
import wandb
import yaml

from nnscaling.scripts.train_mlp_yin_yang.train import main as run_train


def main(sweep_config_path: Path):
    sweep_config = yaml.safe_load(Path(sweep_config_path).read_text())

    # Initialize sweep by passing in config.
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=sweep_config["entity"],
        project=sweep_config["project"],
    )

    # Start sweep job.
    wandb.agent(sweep_id, function=run_train, count=sweep_config["num_sweeps"] or 5)


if __name__ == "__main__":
    fire.Fire(main)
