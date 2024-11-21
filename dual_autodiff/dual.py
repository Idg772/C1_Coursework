import numpy as np

class Dual:

    def __init__(self, real, dual):
        self.real = real
        self.dual = dual
    
#We are using Python's magic methods to define how the Dual object's algebra works

    def __add__(self, other):
        if isinstance(other, Dual): #Dual + Dual
            return Dual(self.real + other.real, self.dual + other.dual)
        elif isinstance(other, (int,float)): #Dual + Real Number
            return Dual(self.real + other, self.dual)
        else:
            return NotImplementedError
    
    def __radd__(self, other): # This just allows us to do c + dual (a,b) = dual(a+c, b) where c is a Real number
        return self.__add__(other)
    
    def __neg__(self):
        return Dual(-self.real, -self.dual)
    
    def __sub__(self, other):
        if isinstance(other, Dual): #Dual - Dual
            return Dual(self.real - other.real, self.dual - other.dual)
        elif isinstance(other, (int,float)): #Dual - Real Number
            return Dual(self.real - other, self.dual)
        else:
            return NotImplementedError
    
    def __rsub__(self, other): # c - dual(a,b) = dual(c-a,-b)
        if isinstance(other, (int, float)):
            return Dual(other - self.real, -self.dual)
        else:
            return NotImplementedError
    
    def __mul__(self, other): # Dual * Dual and Dual * c where c is a real number
        if isinstance(other, Dual):
            return Dual(self.real * other.real, (self.real * other.dual) + (self.dual * other.real))
        elif isinstance(other, (int,float)):
            return Dual(self.real * other, self.dual * other)
        else:
            return NotImplementedError
    
    def __rmul__(self, other): # c * Dual where c is a real number
        return self.__mul__(other)
    
    #NB __div__ is depricated
    def __truediv__(self, other): # Dual/Dual and Dual/c where c is a real number
        if isinstance(other, Dual):
            if other.real == 0:
                raise ZeroDivisionError("No unique solution for division by a Dual number with no real part")
            else:
                return Dual(self.real / other.real, (self.dual*other.real - self.real*other.dual)/(other.real**2))
        elif isinstance(other, (int,float)):
            return Dual(self.real / other, self.dual / other) # Native Python will raise a zero division error here if needed
        else:
            return NotImplementedError
    
    def __rtruediv__(self, other): # c/Dual
        if self.real == 0:
            raise ZeroDivisionError("No unique solution for division by a Dual number with no real part")
        if isinstance(other, (int, float)):
            return Dual(other/self.real, (other * self.dual)/(self.real**2))
        else:
            return NotImplementedError
        
    def __pow__(self, power):
        if isinstance(power,(int,float)): #Dual**real
            if self.real == 0 and power < 0:
                return ZeroDivisionError
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
        if isinstance(other, (int, float)):
            if other <= 0:
                raise ValueError("Cannot raise a non-positive number to a Dual number")
            real_part = other**self.real
            dual_part = real_part * self.dual * np.log(other)
            return Dual(real_part, dual_part)
        else:
            return NotImplementedError
    
    def __repr__(self): #String representation, needed for printing
        return f"Dual(real={self.real}, dual={self.dual})"     
    
    #Below are implementations of standard functions. The expressions come from f(a+be) = f(a) + f'(a)be
    def sin(self):
        real_part = np.sin(self.real)
        dual_part = self.dual * np.cos(self.real)
        return Dual(real_part, dual_part)
    
    def cos(self):
        real_part = np.cos(self.real)
        dual_part = -self.dual * np.sin(self.real)
        return Dual(real_part, dual_part)
    
    def tan(self):
        real_part = np.tan(self.real)
        dual_part = self.dual * (1/np.cos(self.real))**2
        return Dual(real_part, dual_part)
    
    def log(self):
        if self.real <=0:
            raise ValueError("Cannot take the Logarithm of a Dual number with a non-positive real component")
        real_part = np.log(self.real)
        dual_part = self.dual/self.real
        return Dual(real_part, dual_part)
    
    def exp(self):
        real_part = np.exp(self.real)
        dual_part = self.dual * real_part
        return Dual(real_part, dual_part)
    
    #https://numpy.org/devdocs/user/basics.subclassing.html Documentation
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs): #Incomplete
        '''
        Allows Numpyy universal funcuntions to operate on Duals and arrays of Duals
        NB We have not create a custom numpy dtype. It is still dtype=object so other classes
        can be entered into an array. This will likely result in an error when a function is called

        Parameters:
            ufunc: NumPy universal function to apply
            method: String specifing the method name
            *inputs: Dual numbers or arrays containing Dual numbers
            **kwargs: Additional keyword arguments of the ufunc

        Returns:
            Result of the function while maintaining the algebra of the Dual numbers
        
        Raises:
            Not Implemented if method != "__call__" or inputs are invalid types
        '''
        if method != "__call__":
            return NotImplemented
        if not all(isinstance(x, (Dual, np.ndarray)) and 
                       (isinstance(x, Dual) or x.dtype == object) for x in inputs):
                return NotImplemented
        print('inputs')
                
        # Check if first input has the requested operation method (e.g., "sin", "exp")
        # Create a vectorized function that can operate element-wise on arrays
        # Create function that:
        # - Takes arguments as tuple x 
        # - Gets method from first arg (x[0])
        # - Applies method using remaining args (x[1:])
        # Apply vectorized function to first input and remaining inputs
        if hasattr(inputs[0], method):
            return np.vectorize(lambda *x: getattr(x[0], method)(*x[1:]))(inputs[0], *inputs[1:])
        
x = np.array([Dual(2,1), Dual(1,1)])
print(np.exp2(x))
'''
This gives an error because we do not have exp2 defined.
__array_ufunc__ will give the correct vectorized behaviour for the functions we have defined.
TODO: Figure out if there is an easy way of doing this so that we can keep the algebra of the duals
without implementing by hand.
Some ufuncs like the bitwise operators don't really make sense to include.
'''