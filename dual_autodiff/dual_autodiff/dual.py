import numpy as np
from typing import Callable, Union, List

class Dual:
    """
    A class implementing dual numbers for automatic differentiation.
    
    A dual number has the form a + bε where:
    - a is the real part
    - b is the dual part
    - ε is the dual unit with the property ε² = 0
    """

    def __init__(self, real, dual):
        """
        Initialize a dual number.

        Args:
            real: The real part of the dual number
            dual: The dual part of the dual number
        """
        self.real = real
        self.dual = dual
    
    def __add__(self, other):
        """
        Add two dual numbers or a dual number and a scalar.

        Args:
            other: A Dual number or scalar to add

        Returns:
            Dual: The sum of the two numbers
        """
        if isinstance(other, Dual):
            return Dual(self.real + other.real, self.dual + other.dual)
        else:
            return Dual(self.real + other, self.dual)
    
    def __radd__(self, other):
        """
        Reverse addition - called when a non-Dual is added to a Dual.
        """
        return self.__add__(other)
    
    def __sub__(self, other):
        """
        Subtract two dual numbers or a scalar from a dual number.

        Args:
            other: A Dual number or scalar to subtract

        Returns:
            Dual: The difference of the two numbers
        """
        if isinstance(other, Dual):
            return Dual(self.real - other.real, self.dual - other.dual)
        else:
            return Dual(self.real - other, self.dual)
    
    def __rsub__(self, other):
        """
        Reverse subtraction - called when a non-Dual is subtracted from a Dual.
        """
        return self.__sub__(other)

    def __abs__(self):
        """
        Calculate the absolute value of a dual number's real part.

        Returns:
            float: Absolute value of the dual number. Because ε^2=0, this is just the absolute
                   value of the real part.
        """
        return abs(self.real)
    
    def __mul__(self, other):
        """
        Multiply two dual numbers or a dual number and a scalar.

        Uses the product rule: (a + bε)(c + dε) = ac + (ad + bc)ε

        Args:
            other: A Dual number or scalar to multiply

        Returns:
            Dual: The product of the two numbers
        """
        if isinstance(other, Dual):
            return Dual(self.real * other.real, self.real * other.dual + self.dual * other.real)
        else:
            return Dual(self.real * other, self.dual * other)
        
    def __rmul__(self, other):
        """
        Reverse multiplication - called when a non-Dual is multiplied by a Dual.
        """
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """
        Divide two dual numbers or divide a dual number by a scalar.

        Uses the quotient rule for dual numbers.

        Args:
            other: A Dual number or scalar to divide by

        Returns:
            Dual: The quotient of the two numbers

        Raises:
            ZeroDivisionError: If dividing by zero
        """
        if isinstance(other, Dual):
            return Dual(self.real / other.real, 
                       (self.dual * other.real - self.real * other.dual) / other.real**2)
        else:
            return Dual(self.real / other, self.dual / other)
    
    def __rtruediv__(self, other):
        return self.__truediv__(other)
    
    def __pow__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real ** other.real, self.real ** other.real * (self.dual * other.real / self.real + other.dual * np.log(self.real)))
        else:
            return Dual(self.real ** other, other * self.real ** (other - 1) * self.dual)
        
    def __rpow__(self, other):
        return self.__pow__(other)
    
    def __neg__(self):
        return Dual(-self.real, -self.dual)
    
    def __repr__(self):
        return f'Dual({self.real}, {self.dual})'
    
    
    def __eq__(self, other):
        if isinstance(other, Dual):
            return self.real == other.real and self.dual == other.dual
        else:
            return self.real == other and self.dual == 0
        
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def sin(self):
        return Dual(np.sin(self.real), np.cos(self.real) * self.dual)
    
    def cos(self):
        return Dual(np.cos(self.real), -np.sin(self.real) * self.dual)
    
    def tan(self):
        return Dual(np.tan(self.real), self.dual / np.cos(self.real))
    
    def exp(self):
        return Dual(np.exp(self.real), np.exp(self.real) * self.dual)
    
    def log(self):
        return Dual(np.log(self.real), self.dual / self.real)
    
    def sqrt(self):
        return Dual(np.sqrt(self.real), self.dual / (2 * np.sqrt(self.real)))
    
    def sinh(self):
        return Dual(np.sinh(self.real), np.cosh(self.real) * self.dual)
    
    def cosh(self):
        return Dual(np.cosh(self.real), np.sinh(self.real) * self.dual)
    
    def tanh(self):
        return Dual(np.tanh(self.real), self.dual / np.cosh(self.real))
    
    def arcsin(self):
        return Dual(np.arcsin(self.real), self.dual / np.sqrt(1 - self.real**2))
    
    def arccos(self):
        """
        Calculate the inverse cosine (arccos) of a Dual number.

        Returns a Dual number containing arccos(x) in the real part and 
        -1/sqrt(1-x^2) * dx in the dual part, where x is the real part and dx is the dual part.

        Domain: -1 <= real part <= 1
        """
        return Dual(np.arccos(self.real), -self.dual / np.sqrt(1 - self.real**2))
    
    def arctan(self):
        """
        Calculate the inverse tangent (arctan) of a Dual number.

        Returns a Dual number containing arctan(x) in the real part and 
        1/(1+x^2) * dx in the dual part, where x is the real part and dx is the dual part.

        Domain: all real numbers
        """
        return Dual(np.arctan(self.real), self.dual / (1 + self.real**2))
    
    def arcsinh(self):
        """
        Calculate the inverse hyperbolic sine (arcsinh) of a Dual number.

        Returns a Dual number containing arcsinh(x) in the real part and 
        1/sqrt(x^2+1) * dx in the dual part, where x is the real part and dx is the dual part.

        Domain: all real numbers
        """
        return Dual(np.arcsinh(self.real), self.dual / np.sqrt(self.real**2 + 1))
    
    def arccosh(self):
        """
        Calculate the inverse hyperbolic cosine (arccosh) of a Dual number.

        Returns a Dual number containing arccosh(x) in the real part and 
        1/sqrt(x^2-1) * dx in the dual part, where x is the real part and dx is the dual part.

        Domain: real part >= 1
        """
        return Dual(np.arccosh(self.real), self.dual / np.sqrt(self.real**2 - 1))
    
    def arctanh(self):
        """
        Calculate the inverse hyperbolic tangent (arctanh) of a Dual number.

        Returns a Dual number containing arctanh(x) in the real part and 
        1/(1-x^2) * dx in the dual part, where x is the real part and dx is the dual part.

        Domain: -1 < real part < 1
        """
        return Dual(np.arctanh(self.real), self.dual / (1 - self.real**2))


def autodiff(func: Callable, x: Union[np.ndarray, List[float], float], 
            dual_part: float = 1.0) -> Union[np.ndarray, float]:
    """Compute the derivative of a function at given points using dual numbers.
    
    Args:
        func: The function to differentiate. Should accept and work with Dual numbers.
        x: Points at which to evaluate the derivative. Can be a single number or array-like.
        dual_part: The dual part to use for differentiation (default=1.0)
        
    Returns:
        The derivative(s) of the function at the given point(s).
        Returns a numpy array for array-like input, or a float for scalar input.
        
    Examples:
        >>> def f(x): return x**2  # f(x) = x²
        >>> autodiff(f, 2.0)  # Derivative of x² at x=2
        4.0
        >>> autodiff(f, [1.0, 2.0, 3.0])  # Derivatives at multiple points
        array([2., 4., 6.])
    """
    # Convert input to numpy array if it's array-like
    x_array = np.asarray(x)
    scalar_input = np.isscalar(x)
    
    if scalar_input:
        x_array = np.array([x_array])
    
    # Compute derivatives using dual numbers
    derivatives = np.zeros_like(x_array, dtype=float)
    
    for i, x_val in enumerate(x_array):
        # Create dual number and evaluate function
        dual_x = Dual(x_val, dual_part)
        result = func(dual_x)
        
        # Extract derivative from dual part
        if isinstance(result, Dual):
            derivatives[i] = result.dual / dual_part
        else:
            derivatives[i] = 0.0  # Function returned a constant
            
    return derivatives[0] if scalar_input else derivatives

