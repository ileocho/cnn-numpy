import numpy as np
from src.layers.layer import Layer


class Pool(Layer):

    def __init__(self, pool_size, stride):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

        self.n_h, self.n_w, self.n_c = None, None, None
        self.n_h_prev, self.n_w_prev, self.n_c_prev = None, None, None

        self.w = None
        self.b = None

        self.cache = {}

    def init(self, input_dim):
        self.n_h_prev, self.n_w_prev, self.n_c_prev = input_dim
        self.n_h = int((self.n_h_prev - self.pool_size) / self.stride + 1)
        self.n_w = int((self.n_w_prev - self.pool_size) / self.stride + 1)
        self.n_c = self.n_c_prev

    def forward(self, prev, training):

        batch_size = prev.shape[0]

        a = np.zeros((batch_size, self.n_h, self.n_w, self.n_c))

        for i in range(self.n_h):
            v_start = i * self.stride
            v_end = v_start + self.pool_size

            for j in range(self.n_w):
                h_start = j * self.stride
                h_end = h_start + self.pool_size

                prev_slice = prev[:, v_start:v_end, h_start:h_end, :]

                if training:
                    self.cache_max_mask(prev_slice, (i, j))

                a[:, i, j, :] = np.max(prev_slice, axis=(1, 2))

        if training:
            self.cache['prev'] = prev

        return a

    def backward(self, da):
        prev = self.cache['prev']
        batch_size = prev.shape[0]
        da_prev = np.zeros((batch_size, self.n_h_prev, self.n_w_prev, self.n_c_prev))

        for i in range(self.n_h):
            v_start = i * self.stride
            v_end = v_start + self.pool_size

            for j in range(self.n_w):
                h_start = j * self.stride
                h_end = h_start + self.pool_size

                da_prev[:, v_start:v_end, h_start:h_end, :] += da[:, i:i + 1, j:j + 1, :] * self.cache[(i, j)]

        return da_prev, None, None

    def cache_max_mask(self, prev_slice, coords):
        i, j = coords
        mask = np.zeros_like(prev_slice)

        prev_reshape = prev_slice.reshape(prev_slice.shape[0], prev_slice.shape[1] * prev_slice.shape[2],  prev_slice.shape[3])
        idx = np.argmax(prev_reshape, axis=1)

        ax1, ax2 = np.indices((prev_slice.shape[0], prev_slice.shape[3]))
        mask.reshape(mask.shape[0], mask.shape[1] * mask.shape[2], mask.shape[3])[ax1, idx, ax2] = 1
        self.cache[coords] = mask

    def update_params(self, dw, db):
        pass

    def get_params(self):
        pass

    def get_output_dim(self):
        return self.n_h, self.n_w, self.n_c
