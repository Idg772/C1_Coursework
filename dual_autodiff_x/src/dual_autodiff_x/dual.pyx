# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, tan, exp, log, sqrt
from libc.math cimport sinh, cosh, tanh, asin, acos, atan
from libc.math cimport asinh, acosh, atanh, fabs
from typing import Callable, Union, List

np.import_array()

cdef class Dual:
    """
    A class implementing dual numbers for automatic differentiation.
    
    A dual number has the form a + bε where:
    - a is the real part
    - b is the dual part
    - ε is the dual unit with the property ε² = 0
    """
    cdef public double real
    cdef public double dual
    
    def __init__(self, double real, double dual):
        self.real = real
        self.dual = dual
    
    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real + (<Dual>other).real, 
                       self.dual + (<Dual>other).dual)
        else:
            return Dual(self.real + other, self.dual)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real - (<Dual>other).real, 
                       self.dual - (<Dual>other).dual)
        else:
            return Dual(self.real - other, self.dual)
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Dual(other - self.real, -self.dual)
        return NotImplemented

    def __abs__(self):
        return fabs(self.real)
    
    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real * (<Dual>other).real,
                       self.real * (<Dual>other).dual + self.dual * (<Dual>other).real)
        else:
            return Dual(self.real * other, self.dual * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        cdef double other_real, other_dual
        
        if isinstance(other, Dual):
            other_real = (<Dual>other).real
            other_dual = (<Dual>other).dual
            if other_real == 0:
                raise ZeroDivisionError("Division by zero")
            return Dual(self.real / other_real,
                      (self.dual * other_real - self.real * other_dual) / 
                      (other_real * other_real))
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return Dual(self.real / other, self.dual / other)
        else:
            return NotImplemented
    
    def __rtruediv__(self, other):
        if self.real == 0:
            raise ZeroDivisionError("Division by zero")
        if isinstance(other, (int, float)):
            return Dual(other / self.real, 
                       -other * self.dual / (self.real * self.real))
        return NotImplemented
    
    def __pow__(self, other):
        cdef double power_real, power_dual, log_val
        
        if isinstance(other, (int, float)):
            if self.real == 0 and other < 0:
                raise ZeroDivisionError("Cannot raise 0 to negative power")
            power_real = self.real ** other
            power_dual = other * self.dual * (self.real ** (other - 1))
            return Dual(power_real, power_dual)
        
        elif isinstance(other, Dual):
            if self.real <= 0:
                raise ValueError("Base must be positive")
            power_real = self.real ** (<Dual>other).real
            log_val = log(self.real)
            power_dual = (self.dual * (<Dual>other).real * 
                         (self.real ** ((<Dual>other).real - 1))) + \
                        power_real * log_val * (<Dual>other).dual
            return Dual(power_real, power_dual)
        return NotImplemented
    
    def __rpow__(self, other):
        cdef double real_part
        if isinstance(other, (int, float)):
            if other <= 0:
                raise ValueError("Base must be positive")
            real_part = other ** self.real
            return Dual(real_part, real_part * log(other) * self.dual)
        return NotImplemented
    
    def __neg__(self):
        return Dual(-self.real, -self.dual)
    
    def __repr__(self):
        return f'Dual({self.real}, {self.dual})'
    
    def sin(self):
        return Dual(sin(self.real), cos(self.real) * self.dual)
    
    def cos(self):
        return Dual(cos(self.real), -sin(self.real) * self.dual)
    
    def tan(self):
        cdef double cos_val = cos(self.real)
        return Dual(tan(self.real), self.dual / (cos_val * cos_val))
    
    def exp(self):
        cdef double exp_val = exp(self.real)
        return Dual(exp_val, exp_val * self.dual)
    
    def log(self):
        if self.real <= 0:
            raise ValueError("Cannot compute log of non-positive number")
        return Dual(log(self.real), self.dual / self.real)
    
    def sqrt(self):
        cdef double sqrt_val
        if self.real < 0:
            raise ValueError("Cannot compute sqrt of negative number")
        sqrt_val = sqrt(self.real)
        return Dual(sqrt_val, self.dual / (2 * sqrt_val))
    
    def sinh(self):
        return Dual(sinh(self.real), cosh(self.real) * self.dual)
    
    def cosh(self):
        return Dual(cosh(self.real), sinh(self.real) * self.dual)
    
    def tanh(self):
        cdef double cosh_val = cosh(self.real)
        return Dual(tanh(self.real), self.dual / (cosh_val * cosh_val))
    
    def arcsin(self):
        if fabs(self.real) >= 1:
            raise ValueError("arcsin domain error")
        return Dual(asin(self.real), 
                   self.dual / sqrt(1 - self.real * self.real))
    
    def arccos(self):
        if fabs(self.real) >= 1:
            raise ValueError("arccos domain error")
        return Dual(acos(self.real), 
                   -self.dual / sqrt(1 - self.real * self.real))
    
    def arctan(self):
        return Dual(atan(self.real),
                   self.dual / (1 + self.real * self.real))
    
    def arcsinh(self):
        return Dual(asinh(self.real),
                   self.dual / sqrt(self.real * self.real + 1))
    
    def arccosh(self):
        if self.real < 1:
            raise ValueError("arccosh domain error")
        return Dual(acosh(self.real),
                   self.dual / sqrt(self.real * self.real - 1))
    
    def arctanh(self):
        if fabs(self.real) >= 1:
            raise ValueError("arctanh domain error")
        return Dual(atanh(self.real),
                   self.dual / (1 - self.real * self.real))

def autodiff(func: Callable, x: Union[np.ndarray, List[float], float], 
            double dual_part=1.0) -> Union[np.ndarray, float]:
    cdef np.ndarray[double, ndim=1] derivatives
    cdef np.ndarray[double, ndim=1] x_array
    cdef Py_ssize_t i
    cdef double x_val
    cdef object result
    cdef bint scalar_input = np.isscalar(x)
    
    if scalar_input:
        result = func(Dual(float(x), dual_part))
        if isinstance(result, Dual):
            return float((<Dual>result).dual / dual_part)
        return 0.0
        
    x_array = np.asarray(x, dtype=np.float64)
    derivatives = np.zeros_like(x_array, dtype=np.float64)
    
    for i in range(x_array.shape[0]):
        x_val = x_array[i]
        result = func(Dual(x_val, dual_part))
        
        if isinstance(result, Dual):
            derivatives[i] = (<Dual>result).dual / dual_part
        else:
            derivatives[i] = 0.0
            
    return derivatives