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
    
    def exp2(self):
        print("exp2 called directly on Dual")
        real_part = np.exp2(self.real)
        dual_part = self.dual * np.exp2(self.real) * np.log(2)
        return Dual(real_part, dual_part)

    #https://numpy.org/devdocs/user/basics.subclassing.html Documentation
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        print("Called with ufunc:", ufunc)  # See what function is being called
        print("Method:", method)            # Check the method
        print("Inputs:", inputs)            # Look at the inputs
        print("Input types:", [type(x) for x in inputs])  # Check input types
        '''
        Allows Numpy universal functions to operate on Duals and arrays of Duals
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
        
        UNARY_UFUNC_TO_DIFF = {
            np.sin: (np.sin, np.cos),
            np.cos: (np.cos, lambda x: -np.sin(x)),
            np.exp: (np.exp, np.exp),
            np.log: (np.log, lambda x: 1/x),
            np.sqrt: (np.sqrt, lambda x: 0.5/np.sqrt(x)),
            np.cbrt: (np.cbrt, lambda x: 1/(3*np.cbrt(x**2))),
            np.square: (np.square, lambda x: 2*x),
            np.reciprocal: (np.reciprocal, lambda x: -1/x**2),
            np.tan: (np.tan, lambda x: 1/np.cos(x)**2),
            np.arcsin: (np.arcsin, lambda x: 1/np.sqrt(1 - x**2)),
            np.arccos: (np.arccos, lambda x: -1/np.sqrt(1 - x**2)),
            np.arctan: (np.arctan, lambda x: 1/(1 + x**2)),
            np.sinh: (np.sinh, np.cosh),
            np.cosh: (np.cosh, np.sinh),
            np.tanh: (np.tanh, lambda x: 1 - np.tanh(x)**2),
            np.exp2: (np.exp2, lambda x: np.exp2(x) * np.log(2)),
            np.log2: (np.log2, lambda x: 1/(x * np.log(2))),
            np.log10: (np.log10, lambda x: 1/(x * np.log(10)))
        }

        # Dictionary for binary operations: (function, (derivative_wrt_x, derivative_wrt_y))
        BINARY_UFUNC_TO_DIFF = {
            np.add: (np.add, (lambda x,y: 1, lambda x,y: 1)),
            np.subtract: (np.subtract, (lambda x,y: 1, lambda x,y: -1)),
            np.multiply: (np.multiply, (lambda x,y: y, lambda x,y: x)),
            np.divide: (np.divide, (lambda x,y: 1/y, lambda x,y: -x/y**2)),
            np.power: (np.power, (lambda x,y: y * x**(y-1), 
                                lambda x,y: x**y * np.log(x)))
        }

        # Handle numpy arrays first
        if isinstance(inputs[0], np.ndarray):
            print("Input is numpy array")
            if ufunc in UNARY_UFUNC_TO_DIFF:
                print("Found in UNARY_UFUNC_TO_DIFF")
                f, fprime = UNARY_UFUNC_TO_DIFF[ufunc]
                result = []
                for dual in inputs[0]:
                    print("Processing element:", dual)
                    # Apply function and derivative directly
                    real_part = f(dual.real)
                    dual_part = dual.dual * fprime(dual.real)
                    result.append(Dual(real_part, dual_part))
                return np.array(result)
            elif ufunc in BINARY_UFUNC_TO_DIFF and len(inputs) == 2:
                f, (dfdx, dfdy) = BINARY_UFUNC_TO_DIFF[ufunc]
                result = []
                for dual1, dual2 in zip(inputs[0], inputs[1]):
                    # Handle binary operations element-wise
                    x_real = dual1.real if isinstance(dual1, Dual) else dual1
                    y_real = dual2.real if isinstance(dual2, Dual) else dual2
                    x_dual = dual1.dual if isinstance(dual1, Dual) else 0
                    y_dual = dual2.dual if isinstance(dual2, Dual) else 0
                    real_part = f(x_real, y_real)
                    dual_part = x_dual * dfdx(x_real, y_real) + y_dual * dfdy(x_real, y_real)
                    result.append(Dual(real_part, dual_part))
                return np.array(result)

        # Handle unary operations
        if len(inputs) == 1 and ufunc in UNARY_UFUNC_TO_DIFF:
            x = inputs[0]
            f, fprime = UNARY_UFUNC_TO_DIFF[ufunc]
            return Dual(f(x.real), x.dual * fprime(x.real))
        
        # Handle binary operations
        if len(inputs) == 2 and ufunc in BINARY_UFUNC_TO_DIFF:
            x, y = inputs[0], inputs[1]
            f, (dfdx, dfdy) = BINARY_UFUNC_TO_DIFF[ufunc]
            
            # Handle different input types (Dual vs real number)
            x_real = x.real if isinstance(x, Dual) else x
            y_real = y.real if isinstance(y, Dual) else y
            x_dual = x.dual if isinstance(x, Dual) else 0
            y_dual = y.dual if isinstance(y, Dual) else 0
            
            real_part = f(x_real, y_real)
            dual_part = x_dual * dfdx(x_real, y_real) + y_dual * dfdy(x_real, y_real)
            
            return Dual(real_part, dual_part)
                
        # Check if first input has the requested operation method (e.g., "sin", "exp")
        if hasattr(inputs[0], method):
            return np.vectorize(lambda *x: getattr(x[0], method)(*x[1:]))(inputs[0], *inputs[1:])
        
        return NotImplemented


