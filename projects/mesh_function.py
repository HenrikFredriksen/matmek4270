from collections.abc import Callable

import numpy as np


def mesh_function(f: Callable[[float], float], t: np.ndarray) -> np.ndarray:
    return np.array([f(ti) for ti in t])

def func(t: float) -> float:
    if t >= 0 and t <= 3:
        return np.exp(-t)
    elif t > 3 and t <= 4:
        return np.exp(-3 * t)
    
    return 0

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)

if __name__ == "__main__":
    test_mesh_function()
