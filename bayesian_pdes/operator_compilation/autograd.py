from .. import pairwise


class AutogradOperatorSystem(object):
    def __init__(self, operators, operators_bar, kernel, wrapper=None):
        self.kernel = kernel
        self.__operators__ = operators
        self.__operators_bar__ = operators_bar
        self.__cache__ = {}
        self.__wrapper__ = wrapper

    @property
    def operators(self):
        return self.__operators__

    @property
    def operators_bar(self):
        return self.__operators_bar__

    def __getitem__(self, item):
        if item in self.__cache__:
            return self.__cache__[item]

        ret = self.get_raw(item)
        ret = self.pairwiseify(ret)

        self.__cache__[item] = ret
        return ret

    def get_raw(self, item):
        current = self.kernel
        for operator in item:
            if operator not in self.operators and operator not in self.operators_bar:
                raise Exception('Operator {} not valid!')
            current = operator(current)
        return current

    def pairwiseify(self, function):
        def __ret__(x, y, args):
            if self.__wrapper__ is not None:
                return pairwise.apply(self.__wrapper__(function), x, y, args)
            return pairwise.apply(function, x, y, args)
        return __ret__