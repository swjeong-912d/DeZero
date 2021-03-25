# Chapter 10. Test

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from collections import deque

import numpy as np

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Variable:
    def __init__(self, data):
        if data and not isinstance(data, np.ndarray):
            raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
    def clear_grad(self):
        self.grad = None
    def backward(self):
        if self.grad is None:
            # starts from itself, thus gradient is one
            self.grad = np.ones_like(self.data)
        funcs = deque()
        cnt = 0
        funcs = [self.creator]
        # 1 -> N1 -> N1*N2 -> N1*N2*N3 ...
        while funcs:
            cnt += 1
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys) # backward of gys
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            # The set of creator for each input variable
            for x, gx in zip(f.inputs, gxs):
                if x.grad:
                    x.grad = x.grad + gx
                else:
                    x.grad = gx
                if x.creator:
                    funcs.append(x.creator)

class Function():
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    def backward(self, gy):
        return (gy,gy)

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def numerical_diff(f, x, eps = 1e-4):
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def square(x):
    return Square()(x)
def add(x0, x1):
    return Add()(x0, x1)

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


def exp(x):
    return Exp()(x)




def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x0, x1 = Variable(np.array(3)), Variable(np.array(3))
    z = add(x0,x0)
    z.backward() # 2
    x0.clear_grad()
    z2 = add(x0,x0) # 2
    z2.backward()
    print(x0.grad)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
