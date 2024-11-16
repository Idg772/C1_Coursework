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
    
    def __sub__(self, other):
        if isinstance(other, Dual): #Dual - Dual
            return Dual(self.real - other.real, self.dual - other.dual)
        elif isinstance(other, (int,float)): #Dual - Real Number
            return Dual(self.real - other, self.dual)
        else:
            return NotImplementedError
    
    def __rsub__(self, other): # c - dual(a,b) = dual(c-a,-b)
        if isinstance(other, (int, float)):
            return Dual(other - self.real, -self.float)
        else:
            return NotImplementedError
    
    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real * other.real, (self.real * other.dual) + (self.dual + other.real))
        elif isinstance(other, (int,float)):
            return Dual(self.real * other, self.dual * other)
        else:
            return NotImplementedError
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other): #NB __div__ is depricated
        if isinstance(other, Dual):
            if other.real == 0:
                raise ZeroDivisionError("No unique solution for division by a Dual number with no real part")
            else:
                return Dual(self.real / other.real, (self.dual*other.real - self.real*other.dual)/(other.real**2))
        elif isinstance(other, (int,float)):
            return Dual(self.real / other, self.dual / other)
        else:
            return NotImplementedError
    
    def __rtruediv__(self, other):
        if self.real == 0:
            raise ZeroDivisionError("No unique solution for division by a Dual number with no real part")
        if isinstance(other, (int, float)):
            return Dual(other/self.real, (other * self.dual)/(self.real**2))
        else:
            return NotImplementedError
        
            

        