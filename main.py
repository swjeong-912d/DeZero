# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
class Square(Function):
    def forward(self, x):
        return x ** 2
class Exp(Function):
    def forward(self, x):
        return np.exp(x)

def numerical_diff(f, x, eps = 1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data)/(2* eps)

def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x = Variable(np.array(15))
    dy = numerical_diff(f,x)
    t = f(x).data * 4 * x.data
    print ((t-dy)/t)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
