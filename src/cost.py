import numpy as np

epsilon = 1e-20


class CostFunction:
    def f(self, last, y):
        raise NotImplementedError

    def grad(self, last, y):
        raise NotImplementedError


class SigmoidCrossEntropy(CostFunction):
    def f(self, last, y):
        batch_size = y.shape[0]
        last = np.clip(last, epsilon, 1.0 - epsilon)  # min(max)
        cost = -1 / batch_size * np.sum(y * np.log(last) + (1 - y) * np.log(1 - last))
        return cost

    def grad(self, last, y):
        last = np.clip(last, epsilon, 1 - epsilon)  # min(max)
        return -(np.divide(y , last) - np.divide((1 - y), (1 - last)))


class SoftmaxCrossEntropy(CostFunction):
    def f(self, a_last, y):
        batch_size = y.shape[0]
        cost = -1 / batch_size * np.sum(y * np.log(np.clip(a_last, epsilon, 1.0)))
        return cost

    def grad(self, a_last, y):
        return - np.divide(y, np.clip(a_last, epsilon, 1.0))


softmax_cross_entropy = SoftmaxCrossEntropy()
sigmoid_cross_entropy = SigmoidCrossEntropy()
