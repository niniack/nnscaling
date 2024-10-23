from pathlib import Path

import fire
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from safetensors.torch import save_model
from torch import nn

from nnscaling.data import (
    DatasetConfig,
    YinYangBinaryDataset,
    create_data_loader,
)
from nnscaling.log import logger
from nnscaling.models import MLP
from nnscaling.scripts.train_mlp_yin_yang.common import train
from nnscaling.utils import get_device, load_config, set_seed


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: NonNegativeInt = Field(
        0,
        description=(
            "Seed set at start of script. "
            "Also used for train_data.seed and eval_data.seed if they are not set explicitly."
        ),
    )
    train_data: DatasetConfig
    batch_size: PositiveInt
    learning_rate: PositiveFloat
    weight_decay: PositiveFloat
    out_features: PositiveInt
    print_freq: PositiveInt
    save_dir: str
    hidden_neurons: list[PositiveInt]
    num_epochs: PositiveInt | None = None


def main(config_path_or_obj: Path | str | Config):
    config = load_config(config_path_or_obj, config_model=Config)
    device = get_device()

    set_seed(config.seed)
    logger.info(config)

    dataset_config = config.train_data

    # dataset = YinYangNoDotsBinaryDataset(config=dataset_config)
    dataset = YinYangBinaryDataset(config=dataset_config)

    train_loader = create_data_loader(
        dataset, batch_size=config.batch_size, global_seed=config.seed
    )

    model = MLP(
        config=config.hidden_neurons,
        in_features=dataset.features.shape[-1],
        out_features=config.out_features,
        nonlinearity=nn.ReLU,
    )
    model.to(device)
    model.train()

    all_param_names = [name for name, _ in model.named_parameters()]

    assert len(all_param_names) > 0, "No trainable parameters found."
    logger.info(f"Trainable parameters: {len(all_param_names)}")
    model.summary()

    train(
        config=config,
        model=model,
        dataloader=train_loader,
        device=device,
    )

    metadata_dict = {
        "config": str(config.hidden_neurons),
        "dataset": dataset.name(),
        "nonlinearity": "relu",
    }

    save_model(
        model,
        Path(config.save_dir, "yinyang_model.safetensors"),
        metadata=metadata_dict,
    )


if __name__ == "__main__":
    fire.Fire(main)
