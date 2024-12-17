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
    
    def __truediv__(self, other): # Dual/Dual and Dual/c where c is a real number
        '''
        Division of two dual numbers or a dual number and a scalar.
        Uses the quotient rule: (a + bε)/(c + dε) = (a/c) + ((bc - ad)/(c^2))ε  for c != 0
        
        Args:
            other: A Dual number or scalar to divide by
            
        Returns:
            Dual: The quotient of the two numbers
            
        Raises:
            ZeroDivisionError: If the divisor is a Dual number with zero real part
        '''
        if isinstance(other, Dual):
            if other.real == 0:
                raise ZeroDivisionError("No unique solution for division by a Dual number with no real part")
            else:
                return Dual(self.real / other.real, (self.dual*other.real - self.real*other.dual)/(other.real**2))
        elif isinstance(other, (int,float)):
            return Dual(self.real / other, self.dual / other) # Native Python will raise a zero division error here if needed
        else:
            return NotImplementedError
    
    def __rtruediv__(self, other):  # c / Dual(a, b)
        '''
        Division of a scalar by a dual number.
        Uses the quotient rule: c/(a + bε) = (c/a) + (-(bc)/(a^2))ε for a != 0
        
        Args:
            other: A scalar to divide by the dual number
            
        Returns:
            Dual: The quotient of the two numbers
            
        Raises:
            ZeroDivisionError: If the dividend is a scalar with value 0
        '''
        
        if self.real == 0:
            raise ZeroDivisionError("Division by a Dual number with zero real part is undefined.")
        if isinstance(other, (int, float)):
            return Dual(other / self.real, (-other * self.dual) / (self.real ** 2))
        else:
            return NotImplemented
    
    def __pow__(self, power):
        '''
        Exponentiation of a dual number by another dual number or a scalar.
        Uses the chain rule: (a + bε)^c = a^c + c*a^(c-1)*bε
        
        Args:
            power: A Dual number or scalar to raise the dual number to
            
        Returns:
            Dual: The result of exponentiation
            
        Raises:
            ZeroDivisionError: If the real part of the dual number is 0 and the power is negative
            ValueError: If the real part of the dual number is 0 and the power is a Dual number
        '''
        if isinstance(power,(int,float)): #Dual**real
            if self.real == 0 and power < 0:
                raise ZeroDivisionError("Cannot raise a dual number with zero real part to a negative power.")
            dual_part = power * self.dual * (self.real**(power-1))
            return Dual(self.real**power, dual_part)
        
        if isinstance(power, Dual): #Dual**Dual
            if self.real == 0:
                raise ValueError("The power of a Dual is only defined for real component greater than 0")
            real_part = self.real ** power.real
            dual_part = (self.dual * power.real * (self.real **(power.real - 1))) + real_part * np.log(self.real)
            return Dual(real_part, dual_part)
        else:
            return NotImplementedError
    
    def __rpow__(self, other): # c**Dual where c is a real number
        '''
        Exponentiation of a scalar by a dual number.
        Uses the chain rule: c**(a + bε) = c**a + c**a * b * log(c) * ε
        
        Args:
            other: A scalar to raise the dual number to
            
        Returns:
            Dual: The result of exponentiation
            
        Raises:
            ValueError: If the scalar is non-positive
        '''
        if isinstance(other, (int, float)):
            if other <= 0:
                raise ValueError("Cannot raise a non-positive number to a Dual number")
            real_part = other**self.real
            dual_part = real_part * self.dual * np.log(other)
            return Dual(real_part, dual_part)
        else:
            return NotImplemented
    
    def __neg__(self):
        """
        Negate a dual number.
        
        Returns:
            Dual: The negated dual number
        """
        return Dual(-self.real, -self.dual)
    
    def __repr__(self):
        '''
        Return a string representation of the dual number.
        
        Returns:
            str: A string representation of the dual number
        '''
        return f'Dual({self.real}, {self.dual})'
    
    
    def __eq__(self, other):
        '''
        Check if two dual numbers are equal.
        
        Args:
            other: A Dual number or scalar to compare with
            
        Returns:
            bool: True if the real and dual parts of the two numbers are equal, False otherwise
        '''
        if isinstance(other, Dual):
            return self.real == other.real and self.dual == other.dual
        else:
            return TypeError("Cannot compare Dual number with non-Dual number.")
        
    def __ne__(self, other):
        '''
        Check if two dual numbers are not equal.
        
        Args:
            other: A Dual number or scalar to compare with
        
        Returns:
            bool: True if the real and dual parts of the two numbers are not equal, False otherwise
        '''
        return not self.__eq__(other)
    
    def __lt__(self, other):
        return TypeError("Comparison operators are not defined for Dual numbers.")
    
    def __le__(self, other):
        return TypeError("Comparison operators are not defined for Dual numbers.")
    
    def __gt__(self, other):
        return TypeError("Comparison operators are not defined for Dual numbers.")
    
    def __ge__(self, other):
        return TypeError("Comparison operators are not defined for Dual numbers.")
    
    def sin(self):
        '''
        Calculate the sine of a Dual number.
        
        Returns:
            Dual: A Dual number containing sin(x) in the real part and cos(x) * ε in the dual part,
                  where x is the real part and ε is the dual part.
        '''
        return Dual(np.sin(self.real), np.cos(self.real) * self.dual)
    
    def cos(self):
        '''
        Calculate the cosine of a Dual number.
        
        Returns:
            Dual: A Dual number containing cos(x) in the real part and -sin(x) * ε in the dual part,
                  where x is the real part and ε is the dual part.
        '''
        return Dual(np.cos(self.real), -np.sin(self.real) * self.dual)
    
    def tan(self):
        '''
        Calculate the tangent of a Dual number.
        
        Returns:
            Dual: A Dual number containing tan(x) in the real part and sec(x)^2 * ε in the dual part,
                  where x is the real part and ε is the dual part.
        '''
        return Dual(np.tan(self.real), 
                self.dual * (1 / np.cos(self.real)**2))
    
    def exp(self):
        '''
        Calculate the exponential of a Dual number.
        
        Returns:
            Dual: A Dual number containing exp(x) in the real part and exp(x) * ε in the dual part,
                  where x is the real part and ε is the dual part.
        '''
        return Dual(np.exp(self.real), np.exp(self.real) * self.dual)
    
    def log(self):
        '''
        Calculate the natural logarithm of a Dual number.
        
        Returns:
            Dual: A Dual number containing log(x) in the real part and ε / x in the dual part,
                  where x is the real part and ε is the dual part.
        '''
        if self.real <= 0:
            raise ValueError("Natural logarithm is not defined for non-positive numbers")
        return Dual(np.log(self.real), self.dual / self.real)
    
    def sqrt(self):
        '''
        Calculate the square root of a Dual number.
        
        Returns:
            Dual: A Dual number containing sqrt(x) in the real part and ε / (2 * sqrt(x)) in the dual part,
                  where x is the real part and ε is the dual part.
        '''
        if self.real < 0:
            raise ValueError("Square root is not defined for negative numbers")
        return Dual(np.sqrt(self.real), self.dual / (2 * np.sqrt(self.real)))
    
    def sinh(self):
        '''
        Calculate the hyperbolic sine of a Dual number.
        
        Returns:
            Dual: A Dual number containing sinh(x) in the real part and cosh(x) * ε in the dual part,
                  where x is the real part and ε is the dual
        '''
        return Dual(np.sinh(self.real), np.cosh(self.real) * self.dual)
    
    def cosh(self):
        '''
        Calculate the hyperbolic cosine of a Dual number.
        
        Returns:
            Dual: A Dual number containing cosh(x) in the real part and sinh(x) * ε in the dual part,
                  where x is the real part and ε is the dual
        '''
        return Dual(np.cosh(self.real), np.sinh(self.real) * self.dual)
    
    def tanh(self):
        '''
        Calculate the hyperbolic tangent of a Dual number.
        
        Returns:
            Dual: A Dual number containing tanh(x) in the real part and sech(x)^2 * ε in the dual part,
                  where x is the real part and ε is the dual
        '''
        return Dual(np.tanh(self.real),
                self.dual * (1 / np.cosh(self.real)**2))
    
    def arcsin(self):
        '''
        Calculate the inverse sine (arcsin) of a Dual number.
        
        Returns:
            Dual: A Dual number containing arcsin(x) in the real part and 1/sqrt(1-x^2) * ε in the dual part,
                  where x is the real part and ε is the dual part.
            
        Domain: -1 <= real part <= 1
        '''
        if self.real < -1 or self.real > 1:
            raise ValueError("Inverse sine is only defined for real part between -1 and 1")
        return Dual(np.arcsin(self.real), self.dual / np.sqrt(1 - self.real**2))
    
    def arccos(self):
        """
        Calculate the inverse cosine (arccos) of a Dual number.

        Returns a Dual number containing arccos(x) in the real part and 
        -1/sqrt(1-x^2) * ε in the dual part, where x is the real part and ε is the dual part.

        Domain: -1 <= real part <= 1
        """
        if self.real < -1 or self.real > 1:
            raise ValueError("Inverse cosine is only defined for real part between -1 and 1")
        return Dual(np.arccos(self.real), -self.dual / np.sqrt(1 - self.real**2))
    
    def arctan(self):
        """
        Calculate the inverse tangent (arctan) of a Dual number.

        Returns a Dual number containing arctan(x) in the real part and 
        1/(1+x^2) * ε in the dual part, where x is the real part and ε is the dual part.

        Domain: all real numbers
        """
        return Dual(np.arctan(self.real), self.dual / (1 + self.real**2))
    
    def arcsinh(self):
        """
        Calculate the inverse hyperbolic sine (arcsinh) of a Dual number.

        Returns a Dual number containing arcsinh(x) in the real part and 
        1/sqrt(x^2+1) * ε in the dual part, where x is the real part and ε is the dual part.

        Domain: all real numbers
        """
        return Dual(np.arcsinh(self.real), self.dual / np.sqrt(self.real**2 + 1))
    
    def arccosh(self):
        """
        Calculate the inverse hyperbolic cosine (arccosh) of a Dual number.

        Returns a Dual number containing arccosh(x) in the real part and 
        1/sqrt(x^2-1) * ε in the dual part, where x is the real part and ε is the dual part.

        Domain: real part >= 1
        """
        if self.real < 1:
            raise ValueError("Inverse hyperbolic cosine is only defined for real part greater than or equal to 1")
        return Dual(np.arccosh(self.real), self.dual / np.sqrt(self.real**2 - 1))
    
    def arctanh(self):
        """
        Calculate the inverse hyperbolic tangent (arctanh) of a Dual number.

        Returns a Dual number containing arctanh(x) in the real part and 
        1/(1-x^2) * ε in the dual part, where x is the real part and ε is the dual part.

        Domain: -1 < real part < 1
        """
        if self.real <= -1 or self.real >= 1:
            raise ValueError("Inverse hyperbolic tangent is only defined for real part between -1 and 1")
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

