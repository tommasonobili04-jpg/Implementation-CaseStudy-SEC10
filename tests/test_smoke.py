from src.marginals import example_density
import numpy as np

def test_example_density():
    x = np.array([0.0])
    y = example_density(x)
    assert y.shape == (1,)
    assert y[0] > 0
