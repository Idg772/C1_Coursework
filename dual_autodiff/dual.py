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
            dual_part = (self.dual * power.real * (self.real **(power.real - 1))) + real_part * np.log(self.real_part)
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
    def ___array_ufunc__(self, ufunc, method, *inputs, **kwargs): #Allows numpy universal functions to be used
        if method != "__call__":
            return NotImplemented
        
        ufunc_map = {
            np.add: self.__add__,
            np.subtract: self.__sub__,
            np.multiply: self.__mul__,
            np.divide: self.__truediv__,
            np.power: self.__pow__,
            np.sin: self.sin,
            np.cos: self.cos,
            np.tan: self.tan,
            np.exp: self.exp,
            np.log: self.log
        }

        if ufunc in ufunc_map:
            func = ufunc_map[ufunc]
            return func(*inputs[1:])
        else:
            return NotImplemented



# x = Dual(1,2)
# print(np.exp(x))
# print(x.log() - np.log(x))
#Note that this produces the correct result of Dual(0,0).
#This is true even though we have implemented __array_ufunc__ because of a legacy feature of NumPy.

x = np.array([Dual(1,1), Dual(1,0)])

print(np.sin(np.sum(x)))
print(type(x))
