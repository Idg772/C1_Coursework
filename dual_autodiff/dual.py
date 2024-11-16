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
    
