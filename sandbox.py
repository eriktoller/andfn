"""
Notes
-----
This is a sandbox module for testing new functions.
"""
import numpy as np

def func_a(var1, var2):

    return var1 + var2


class A:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def func_a(self, var1):
        return var1 + self.a

    def check(self, var1):
        self.b.checkis(1,self)
        print(self.func_a(var1))

class B:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def func_b(self, var1):
        return var1 + self.b

    def checkis(self, var1, var2):
        aa = var1 + var2
        print(self.func_b(var1))

def well(chi, q):
    return q / (2 * np.pi) * np.log(chi)

if __name__ == '__main__':


    i = np.zeros([1,10], dtype=complex)
    ii = np.zeros(10, dtype=complex)
    b = B(1, 2)
    a = A(3, b)

    a.check(1)

    print(a.func_a(1))