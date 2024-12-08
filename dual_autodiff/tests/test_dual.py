import pytest
import numpy as np
from dual_autodiff import Dual, autodiff  

def test_dual_initialization():
    dual = Dual(2.0, 3.0)
    assert dual.real == 2.0
    assert dual.dual == 3.0

def test_dual_addition():
    d1 = Dual(2.0, 3.0)
    d2 = Dual(1.0, 4.0)
    result = d1 + d2
    assert result == Dual(3.0, 7.0)

    result = d1 + 5.0
    assert result == Dual(7.0, 3.0)

def test_dual_subtraction():
    d1 = Dual(5.0, 3.0)
    d2 = Dual(2.0, 1.0)
    result = d1 - d2
    assert result == Dual(3.0, 2.0)

    result = d1 - 1.0
    assert result == Dual(4.0, 3.0)

def test_dual_multiplication():
    d1 = Dual(2.0, 3.0)
    d2 = Dual(4.0, 5.0)
    result = d1 * d2
    assert result == Dual(8.0, 22.0)

    result = d1 * 3.0
    assert result == Dual(6.0, 9.0)

def test_dual_division():
    d1 = Dual(6.0, 3.0)
    d2 = Dual(2.0, 1.0)
    result = d1 / d2
    assert result == Dual(3.0, 0)

    result = d1 / 2.0
    assert result == Dual(3.0, 1.5)

def test_scalar_division_by_dual():
    d = Dual(2.0, 3.0)
    result = 6.0 / d
    assert result == Dual(3.0, -4.5)

def test_dual_power():
    d1 = Dual(2.0, 3.0)
    result = d1 ** 2
    assert result == Dual(4.0, 12.0)

    result = 3.0 ** d1
    assert np.isclose(result.real, 9.0)
    assert np.isclose(result.dual, 9.0 * np.log(3.0) * 3.0)

def test_autodiff_single_point():
    def f(x):
        return x**2 + 3*x + 5

    derivative = autodiff(f, 2.0)
    assert np.isclose(derivative, 7.0)

def test_division_by_zero_real_part():
    d1 = Dual(1.0, 2.0)
    d2 = Dual(0.0, 1.0)
    with pytest.raises(ZeroDivisionError):
        _ = d1 / d2

def test_zero_power_negative_dual():
    d = Dual(0.0, 1.0)
    with pytest.raises(ZeroDivisionError):
        _ = d ** -1

def test_addition_commutativity():
    d1 = Dual(1.0, 2.0)
    d2 = Dual(3.0, 4.0)
    assert d1 + d2 == d2 + d1

def test_multiplication_commutativity():
    d1 = Dual(1.0, 2.0)
    d2 = Dual(3.0, 4.0)
    assert d1 * d2 == d2 * d1


def test_autodiff_multiple_points():
    def f(x):
        return x**2

    derivatives = autodiff(f, [1.0, 2.0, 3.0])
    assert np.allclose(derivatives, [2.0, 4.0, 6.0])

def test_dual_trigonometric():
    d = Dual(np.pi / 4, 1.0)

    sin_d = d.sin()
    assert np.isclose(sin_d.real, np.sin(np.pi / 4))
    assert np.isclose(sin_d.dual, np.cos(np.pi / 4))

    cos_d = d.cos()
    assert np.isclose(cos_d.real, np.cos(np.pi / 4))
    assert np.isclose(cos_d.dual, -np.sin(np.pi / 4))

    tan_d = d.tan()
    assert np.isclose(tan_d.real, np.tan(np.pi / 4))
    assert np.isclose(tan_d.dual, 1 / np.cos(np.pi / 4)**2)

def test_dual_exponential_and_logarithmic():
    d = Dual(2.0, 1.0)

    exp_d = d.exp()
    assert np.isclose(exp_d.real, np.exp(2.0))
    assert np.isclose(exp_d.dual, np.exp(2.0))

    log_d = d.log()
    assert np.isclose(log_d.real, np.log(2.0))
    assert np.isclose(log_d.dual, 0.5)

def test_dual_hyperbolic():
    d = Dual(1.0, 1.0)

    sinh_d = d.sinh()
    assert np.isclose(sinh_d.real, np.sinh(1.0))
    assert np.isclose(sinh_d.dual, np.cosh(1.0))

    cosh_d = d.cosh()
    assert np.isclose(cosh_d.real, np.cosh(1.0))
    assert np.isclose(cosh_d.dual, np.sinh(1.0))

    tanh_d = d.tanh()
    assert np.isclose(tanh_d.real, np.tanh(1.0))
    assert np.isclose(tanh_d.dual, 1 / np.cosh(1.0)**2)
