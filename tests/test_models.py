import torch
import torch.nn as nn

from nnscaling.models import MLP
from nnscaling.models.factory import LinearLayer, LinearLayerFactory
from nnscaling.models.scaler import ScaledModel


def test_layer_init():
    layer_a = LinearLayer(
        in_features=5,
        out_features=2,
        nonlinearity=nn.ReLU,
        bias=True,
    )

    layer_factory = LinearLayerFactory()
    layer_b = layer_factory.create_layer(
        in_features=5,
        out_features=2,
        nonlinearity=nn.ReLU,
        bias=True,
    )

    assert layer_a.linear.weight.shape == layer_b.linear.weight.shape
    assert type(layer_a) is type(layer_b)


def test_mlp_init():
    model = MLP(config=[8, 5], in_features=16, out_features=2, nonlinearity=nn.ReLU)
    # NOTE: Torch weights are transposed
    assert model.features[0].linear.weight.shape == torch.Size((8, 16))
    assert model.features[-1].linear.weight.shape == torch.Size((2, 5))

    input_tensor = torch.rand((1, 16))
    output = model(input_tensor)
    assert output.shape == torch.Size((1, 2))


def test_scaler_init():
    model = MLP(config=[8, 5], in_features=16, out_features=2, nonlinearity=nn.ReLU)
    scaled_model = ScaledModel(model=model, index=1, num_scaled=3)

    assert len(list(scaled_model.children())) == 3

    input_tensor = torch.rand((1, 16))
    output = model.forward(input_tensor)
    scaled_raw_output = scaled_model.forward_raw(input_tensor)

    torch.testing.assert_close(output, scaled_raw_output, atol=0, rtol=0)

    scaled_output = scaled_model.forward(input_tensor)
    assert scaled_output.shape == torch.Size((1, 2))
