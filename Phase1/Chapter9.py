# Chapter 9. Make function easier

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
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

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        f = self.creator
        while f:
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            f = x.creator


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
def square(x):
    return Square()(x)

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
def exp(x):
    return Exp()(x)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(type(np.array(1.0)))
    print(type(np.array(1.0)*1))
    print(type(np.array([1.0])))
    print(type(np.array([1.0])*1))
    # 2번째가 float64로 바뀌는거 땜에 backward 에러가 뜸. 그걸 방지하기 위해서 as_array로 ouput을 감싸줌.
    # Function class의 __call__에 사용

    x = Variable(None)
    x = Variable(np.array(1.0))
    y = square(exp(square(x)))
    assert y.creator.input.creator.input.creator.input == x
    #
    # y.grad = np.array(1.0)
    # b.grad = y.creator.backward(y.grad)
    # a.grad = y.creator.input.creator.backward(b.grad)
    # x.grad = y.creator.input.creator.input.creator.backward(a.grad)
    y.backward()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
