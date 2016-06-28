class CompositeOperatorSystem(object):
    def __init__(self, ops, ops_bar, old_posterior):
        self.__ops__ = ops
        self.__ops_bar__ = ops_bar
        self.__old_posteror__ = old_posterior

    @property
    def operators(self):
        return self.__ops__

    @property
    def operators_bar(self):
        return self.__ops_bar__

    def __getitem__(self, item):
        Identity = ()
        if type(item) is not tuple:
            item = (item,)

        ops = [o for o in item if o in self.operators and o not in self.operators_bar]
        ops_bar = [o for o in item if o in self.operators_bar]

        if len(ops) == 0:
            op = Identity
        elif len(ops) == 1:
            op = ops[0]
        else:
            raise Exception('Only support application of a single operator; received {}'.format(ops))

        if len(ops_bar) == 0:
            op_bar = Identity
        elif len(ops_bar) == 1:
            op_bar = ops_bar[0]
        else:
            raise Exception('Only support application of a single operator; received {}'.format(ops_bar))

        operated_posterior = self.__old_posteror__.apply_operators([op], [op_bar])

        return operated_posterior.kern