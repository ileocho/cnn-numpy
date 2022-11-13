import numpy as np
from src.layers.layer import Layer

class Dropout(Layer):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob <= 1, 'probability of not droppping must be between 0 and 1'
        self.prob = prob
        self.mask_dim = None
        self.cached_mask = None

    def init(self, in_dim):
        self.mask_dim = in_dim

    def forward(self, prev, training):
        if training:
            mask = (np.random.rand(*prev.shape) < self.prob)
            a = self.inverted_dropout(prev, mask)
            self.cached_mask = mask

            return a

        return prev

    def backward(self, da):
        return self.inverted_dropout(da, self.cached_mask), None, None

    def update_params(self, dw, db):
        pass

    def get_params(self):
        pass

    def get_output_dim(self):
        return self.mask_dim

    def inverted_dropout(self, x, mask):
        x *= mask
        x /= self.prob
        return x