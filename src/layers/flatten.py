from src.layers.layer import Layer

class Flatten(Layer):

    def __init__(self):
        super().__init__()
        self.original_dim = None
        self.output_dim = None

    def init(self, input_dim):
        self.original_dim = input_dim
        self.output_dim = self.cumul_func(lambda x, y: x * y, self.original_dim)


    def cumul_func(self, func, iterable, init=None):
        it = iter(iterable)
        if init is not None:
            value = init
        else:
            value = next(it)
        for element in it:
            value = func(value, element)
        return value

    def forward(self, prev, training):
        return prev.reshape(prev.shape[0], -1)

    def backward(self, da):
        return da.reshape(da.shape[0], *self.original_dim), None, None

    def get_params(self):
        pass

    def update_params(self, dw, db):
        pass

    def get_output_dim(self):
        return self.output_dim