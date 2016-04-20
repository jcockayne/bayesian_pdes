class NamedLambda(object):
    def __init__(self, func, desc):
        self.__func__ = func
        self.__desc__ = desc

    def __call__(self, *args):
        return self.__func__(*args)

    def __str__(self):
        return '<NamedLambda: {}>'.format(self.__desc__)

    def __repr__(self):
        return '<NamedLambda: {}>'.format(self.__desc__)