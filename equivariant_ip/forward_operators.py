import abc


class ForwardOperator(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x):
        return

    @abc.abstractmethod
    def vec_jac_prod(self, x, z):
        return

    @abc.abstractmethod
    def pseudoinverse(self, y):
        return
