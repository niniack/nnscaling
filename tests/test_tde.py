import numpy as np
import pytest
from matrepr import mprint
from pykoopman.observables import TimeDelay

from nnscaling.models import TorchDelayEmbedder


def test_tde_simple():
    delay = 1
    n_delays = 2
    pktde = TimeDelay(delay=delay, n_delays=n_delays)
    mytde = TorchDelayEmbedder(delay=delay, n_delays=n_delays)

    # x = np.random.rand(3, 100)
    x = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

    pktde_phi = pktde.fit_transform(x.T).T

    mytde.fit(x)
    mytde_phi = mytde(x).numpy()

    np.testing.assert_array_almost_equal_nulp(pktde_phi, mytde_phi)

    pktde_x = pktde.measurement_matrix_ @ pktde_phi
    mytde_x = mytde.measurement_matrix_ @ mytde_phi

    np.testing.assert_array_almost_equal_nulp(x[:, -1], pktde_x[:, -1])
    np.testing.assert_array_almost_equal_nulp(x[:, -1], mytde_x[:, -1].numpy())


@pytest.mark.parametrize("delay", [1, 5, 10])
@pytest.mark.parametrize("n_delays", [2, 5, 9])
def test_tde_random(delay, n_delays):
    pktde = TimeDelay(delay=delay, n_delays=n_delays)
    mytde = TorchDelayEmbedder(delay=delay, n_delays=n_delays)

    x = np.random.rand(3, 100)

    pktde_phi = pktde.fit_transform(x.T).T

    mytde.fit(x)
    mytde_phi = mytde(x).numpy()

    np.testing.assert_array_almost_equal_nulp(pktde_phi, mytde_phi)

    pktde_x = pktde.measurement_matrix_ @ pktde_phi
    mytde_x = mytde.measurement_matrix_ @ mytde_phi

    np.testing.assert_array_almost_equal_nulp(x[:, -1], pktde_x[:, -1])
    np.testing.assert_array_almost_equal_nulp(x[:, -1], mytde_x[:, -1].numpy())
