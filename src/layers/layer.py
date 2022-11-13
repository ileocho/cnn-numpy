class Layer:

    def init(self, input_dim):
        raise NotImplementedError

    def forward(self, prev, training):
        raise NotImplementedError

    def backward(self, da):
        raise NotImplementedError

    def update_params(self, dw, db):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def get_output_dim(self):
        raise NotImplementedError