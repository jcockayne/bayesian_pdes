import numpy as np
import hashlib
class CachingOpCache(object):
    def __init__(self, base_op_cache):
        self.__base_op_cache__ = base_op_cache
        self.__caches__ = {}

    @property
    def operators(self):
        return self.__base_op_cache__.operators

    @property
    def operators_bar(self):
        return self.__base_op_cache__.operators_bar

    def __getitem__(self, item):
        if item in self.__caches__:
            return self.__caches__[item]
        function = self.__base_op_cache__[item]
        cache_function = FunctionCache(function)
        self.__caches__[item] = cache_function
        return cache_function

    def clear(self):
        self.__caches__ = {}


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
            item = np.ascontiguousarray(self.__array__).view(np.uint8)
            digest = hashlib.sha1(item).hexdigest()
            self.__hash_value__ = int(digest, 16)
        return self.__hash_value__

    def __eq__(self, other):
        if type(other) is not HashableNumpyArray:
            return False

        if other.__array__ is self.__array__:
            return True
        return np.array_equal(self.__array__, other.__array__)