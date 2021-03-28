import heapq
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
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    def clear_grad(self):
        self.grad = None
    def backward(self):
        if self.grad is None:
            # starts from itself, thus gradient is one
            self.grad = np.ones_like(self.data)
        cnt = 0
        funcs = []
        # (variable gen, creator gen, creator input order, creator). tips by python documentation
        heapq.heappush(funcs,(-self.generation, id(self), self.creator))
        while funcs:
            cnt += 1
            vgen, vid, f = heapq.heappop(funcs)
            print('----')
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys) # backward of gys
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            # The set of creator for each input variable
            for i, (x, gx) in enumerate(zip(f.inputs, gxs)):
                print(x.generation)
                if x.grad:
                    x.grad = x.grad + gx
                else:
                    x.grad = gx
                    if x.creator:
                        heapq.heappush(funcs,(-x.generation, id(x), x.creator))
            print(funcs)

class Function():
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        self.generation = max([x.generation for x in inputs])
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
    y0 = square(x0)
    y1 = square(x1)
    y00 = square(y0)
    y11 = square(y1)
    z = add(square(x0),square(x0))
    z.backward()

    print(x0.grad)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
