import pytest
import torch
import torch.nn as nn

from nnscaling.models import MLP
from nnscaling.models.factory import LinearLayer, LinearLayerFactory
from nnscaling.models.scaler import ScaledModel


# Arrange
@pytest.fixture
def model(request):
    return MLP(
        config=request.param, in_features=16, out_features=2, nonlinearity=nn.ReLU
    )


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


@pytest.mark.parametrize("model", [[8, 5]], indirect=True)
def test_mlp_init(model):
    # NOTE: Torch weights are transposed
    assert model.features[0].linear.weight.shape == torch.Size((8, 16))
    assert model.features[-1].linear.weight.shape == torch.Size((2, 5))

    input_tensor = torch.rand((1, 16))
    output = model(input_tensor)
    assert output.shape == torch.Size((1, 2))


@pytest.mark.parametrize("model", [[8, 5]], indirect=True)
def test_scaler_init(model):
    scaled_model = ScaledModel(model=model, index=1, num_scaled=3)

    assert len(list(scaled_model.children())) == 3

    input_tensor = torch.rand((1, 16))
    output = model.forward(input_tensor)
    raw_output_scaled_model = scaled_model.forward_raw(input_tensor)

    torch.testing.assert_close(output, raw_output_scaled_model, atol=0, rtol=0)

    output_scaled_model = scaled_model.forward_scaled(input_tensor)
    assert output_scaled_model.shape == torch.Size((1, 2))


@pytest.mark.parametrize("model", [[10, 8, 5]], indirect=True)
def test_scaler_all_hooks(model):
    num_scaled = 3
    scaled_model = ScaledModel(model=model, index=2, num_scaled=num_scaled)

    assert len(scaled_model.hooks) == 0

    scaled_model.hook_model(pre=True, scaled=True, post=True)

    _ = scaled_model.forward_scaled(torch.rand((1, 16)))

    # Check internals
    assert (
        len(scaled_model.pre_inserts[-1]._forward_hooks)
        == len(scaled_model.post_inserts[0]._forward_hooks)
        == 1
    )
    # Check custom
    assert (
        len(scaled_model.hooks) == len(scaled_model.activations_dict) == num_scaled + 2
    )

    scaled_model.clear_hooks()
    assert (
        len(scaled_model.post_inserts[0]._forward_hooks) == len(scaled_model.hooks) == 0
    )


@pytest.mark.parametrize("model", [[10, 8, 5]], indirect=True)
def test_scaler_pre_and_post_hooks(model):
    num_scaled = 3
    scaled_model = ScaledModel(model=model, index=2, num_scaled=num_scaled)

    assert len(scaled_model.hooks) == 0

    scaled_model.hook_model(pre=True, scaled=False, post=True)

    _ = scaled_model.forward_scaled(torch.rand((1, 16)))

    # Check internals
    assert (
        len(scaled_model.pre_inserts[-1]._forward_hooks)
        == len(scaled_model.post_inserts[0]._forward_hooks)
        == 1
    )
    # Check custom
    assert len(scaled_model.hooks) == len(scaled_model.activations_dict) == 2

    scaled_model.clear_hooks()
    assert (
        len(scaled_model.post_inserts[0]._forward_hooks) == len(scaled_model.hooks) == 0
    )
