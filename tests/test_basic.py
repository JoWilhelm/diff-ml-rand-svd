import jax.random as jrandom

import diff_ml as dml
from diff_ml.model import Bachelier, BachelierParams


# import diff_ml.model


def test_basic():
    assert True


def test_basic2():
    assert dml.test()


def test_basic3():
    assert dml.mol() == 42


def test_basic4():
    key = jrandom.PRNGKey(0)
    b = Bachelier(key)
    assert b.n_dims == 1
