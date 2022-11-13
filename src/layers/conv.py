import numpy as np
from src.activation import identity
from src.layers.layer import Layer


class Conv(Layer):
    """
    This class represents a convolutional layer in a neural network.
    The volume of output is (n_h, n_w, n_c). The "volume" word is used because of batch method, making "cubic" data structures.
    The forward function computes the output of the convolutional layer given the input to that layer.
    The backward function computes the gradient of the loss with respect to the parameters (weights and biases) of the layer. It also computes the gradient of loss with respect to any output pre-activations.
    The update_params function updates the parameters of the layer according to the gradient computed in the backward function.
    The get_params function returns the current values of the layer's parameters.
    """

    def __init__(self, kernel_size, stride, n_c, padding='valid', activation=identity):

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation

        self.n_h = None
        self.n_w = None
        self.n_c = n_c

        self.n_h_prev = None
        self.n_w_prev = None
        self.n_c_prev = None

        self.w = None
        self.b = None

        self.pad = None

        self.cache = {}

    def init(self, input_dim):
        self.pad = 0 if self.padding == 'valid' else int((self.kernel_size - 1) / 2)

        self.n_h_prev, self.n_w_prev, self.n_c_prev = input_dim

        self.n_h = int((self.n_h_prev - self.kernel_size + 2 * self.pad) / self.stride + 1)
        self.n_w = int((self.n_w_prev - self.kernel_size + 2 * self.pad) / self.stride + 1)

        self.w = np.random.randn(self.kernel_size, self.kernel_size, self.n_c_prev, self.n_c)
        self.b = np.zeros((1, 1, 1, self.n_c))

    def forward(self, prev, training):
        """
        The forward function computes the output of the convolutional layer given the input to that layer.
        The function takes as input a 4-dimensional array (batch size, height, width, and channel) representing
        the previous layer , and produces an outpu of the same dimensions representing this l-th layer.
        """
        batch_size = prev.shape[0]
        prev_padded = Conv.zero_pad(prev, self.pad)
        out = np.zeros((batch_size, self.n_h, self.n_w, self.n_c))

        for i in range(self.n_h):
            v_start = i * self.stride
            v_end = v_start + self.kernel_size

            for j in range(self.n_w):
                h_start = j * self.stride
                h_end = h_start + self.kernel_size

                out[:, i, j, :] = np.sum(prev_padded[:, v_start:v_end, h_start:h_end, :, np.newaxis] *
                                         self.w[np.newaxis, :, :, :], axis=(1, 2, 3))

        z = out + self.b
        a = self.activation.f(z)

        if training:
            self.cache.update({'prev': prev, 'z': z, 'a': a})

        return a

    def backward(self, da):
        """
        The backward function computes the gradient of the loss with respect to
        the parameters (weights and biases) of the layer. It also computes the
        gradient of loss with respect to ay output pre-activations. This function
        takes da as an argument which is equal to dL/da computed in forward().
        """

        batch_size = da.shape[0]
        prev, z, a = (self.cache[key] for key in ('prev', 'z', 'a'))
        prev_pad = Conv.zero_pad(prev, self.pad) if self.pad != 0 else prev

        da_prev = np.zeros((batch_size, self.n_h_prev, self.n_w_prev, self.n_c_prev))
        da_prev_pad = Conv.zero_pad(da_prev, self.pad) if self.pad != 0 else da_prev

        dz = da * self.activation.df(z, cached_y=a)
        db = 1 / batch_size * dz.sum(axis=(0, 1, 2))
        dw = np.zeros((self.kernel_size, self.kernel_size, self.n_c_prev, self.n_c))

        for i in range(self.n_h):
            v_start = self.stride * i
            v_end = v_start + self.kernel_size

            for j in range(self.n_w):
                h_start = self.stride * j
                h_end = h_start + self.kernel_size

                da_prev_pad[:, v_start:v_end, h_start:h_end, :] += np.sum(self.w[np.newaxis, :, :, :, :] * dz[:, i:i+1, j:j+1, np.newaxis, :], axis=4)

                dw += np.sum(prev_pad[:, v_start:v_end, h_start:h_end, :, np.newaxis] *
                             dz[:, i:i+1, j:j+1, np.newaxis, :], axis=0)

        dw /= batch_size

        if self.pad != 0:
            da_prev = da_prev_pad[:, self.pad:-self.pad, self.pad:-self.pad, :]

        return da_prev, dw, db

    def get_output_dim(self):
        return self.n_h, self.n_w, self.n_c

    def update_params(self, dw, db):
        self.w -= dw
        self.b -= db

    def get_params(self):
        return self.w, self.b

    @staticmethod
    def zero_pad(x, pad):
        return np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')
