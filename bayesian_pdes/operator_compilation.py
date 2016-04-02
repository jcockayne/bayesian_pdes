from sympy_helpers import sympy_function, n_arg_applier
import pairwise
import numpy as np
import hashlib


def compile_sympy(operators, operators_bar, k, symbols, mode=None, sympy_function_kwargs=None, debug=False):
    ret = {
        tuple(): __functionize(k, symbols, mode=mode, sympy_function_kwargs=sympy_function_kwargs)
    }
    for op in operators:
        if op not in ret:
            ret[(op,)] = __functionize(op(k), symbols, mode=mode, sympy_function_kwargs=sympy_function_kwargs)
    for op in operators_bar:
        if op not in ret:
            ret[(op,)] = __functionize(op(k), symbols, mode=mode, sympy_function_kwargs=sympy_function_kwargs)
    # combinations
    for op in operators:
        for op_bar in operators_bar:
            # don't do anything if already there
            if (op, op_bar) in ret:
                pass
            # exploit symmetry
            elif (op_bar, op) in ret:
                ret[(op, op_bar)] = ret[(op_bar, op)]
            # no choice!!
            else:
                ret[(op, op_bar)] = __functionize(op(op_bar(k)), symbols, mode=mode, sympy_function_kwargs=sympy_function_kwargs)
    return ret


# for now we only support sympy, maybe later support Theano?
# nest inside a function which will apply the result pairwise
def __functionize(fun, symbols, mode=None, apply_factory=n_arg_applier, sympy_function_kwargs=None):
    if sympy_function_kwargs is None:
        sympy_function_kwargs = {}
    sympy_fun = sympy_function(fun, symbols, mode=mode, apply_factory=apply_factory, **sympy_function_kwargs)

    def __ret_function(a, b, extra=None):
        return pairwise.apply(sympy_fun, a, b, extra)

    return __ret_function


class CachingOpCache(object):
    def __init__(self, base_op_cache):
        self.__base_op_cache__ = base_op_cache
        self.__caches__ = {}

    def __getitem__(self, item):
        if item in self.__caches__:
            return self.__caches__[item]
        function = self.__base_op_cache__[item]
        cache_function = FunctionCache(function)
        self.__caches__[item] = cache_function
        return cache_function


def make_args_hashable(*args, **kwargs):
    args = list(args)

    def __convert(item):
        if type(item) is np.ndarray:
            return HashableNumpyArray(item)
        return item

    key = tuple([__convert(a) for a in args])
    return key


class FunctionCache(object):
    def __init__(self, base_function):
        self.__base_function__ = base_function
        self.__arg_cache__ = {}

    def __call__(self, *args, **kwargs):
        hashable = make_args_hashable(*args)
        if hashable in self.__arg_cache__:
            return self.__arg_cache__[hashable]
        to_hash = self.__base_function__(*args)
        self.__arg_cache__[hashable] = to_hash
        return to_hash


class HashableNumpyArray(object):
    def __init__(self, array):
        self.__array__ = array
        self.__hash_value__ = None

    def __hash__(self):
        if self.__hash_value__ is None:
            item = self.__array__.view(np.uint8)
            digest = hashlib.sha1(item).hexdigest()
            self.__hash_value__ = int(digest, 16)
        return self.__hash_value__

    def __eq__(self, other):
        if type(other) is not HashableNumpyArray:
            return False

        if other.__array__ is self.__array__:
            return True
        return np.array_equal(self.__array__, other.__array__)

