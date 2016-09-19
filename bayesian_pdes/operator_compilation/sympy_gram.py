import random, string
import os
import sys
import shutil
from subprocess import STDOUT, CalledProcessError, check_output
import sympy as sp
import tempfile
import re
import compilation_utils

__HEADER__ = """cimport numpy as np
import numpy as np
from libc.math cimport *
from cython.parallel import parallel, prange
"""
__GRAM_EXT__ = '_gram'
__PAR_EXT__ = '_par'

__CONDITION__ = """
if({}):
    return {}
"""

__ROUTINE__ = """# =======================
# {comment}
# =======================
cdef double {name}({args}) nogil:
{impl}

def {name}_raw({args}):
    return {name}({arg_names})

def {name}{gram_ext}(np.ndarray[ndim=2, dtype=np.float_t] xarr, np.ndarray[ndim=2, dtype=np.float_t] yarr, np.ndarray[ndim=1, dtype=np.float_t] fun_args=None):
    if fun_args is None:
        fun_args = np.array([])

{definitions_and_assignments}

    cdef np.ndarray[ndim=2, dtype=np.float_t] ret
    cdef int i,j
    ret = np.empty((xarr.shape[0], yarr.shape[0]))
    for i in xrange(xarr.shape[0]):
        for j in xrange(yarr.shape[0]):
{in_loop_assigns}
            ret[i,j] = {name}({arg_names})
    return ret

def {name}{par_ext}(np.ndarray[ndim=2, dtype=np.float_t] xarr, np.ndarray[ndim=2, dtype=np.float_t] yarr, np.ndarray[ndim=1, dtype=np.float_t] fun_args=None):
    if fun_args is None:
        fun_args = np.array([])

{definitions_and_assignments}

    cdef np.ndarray[ndim=2, dtype=np.float_t] ret
    cdef int i,j
    ret = np.empty((xarr.shape[0], yarr.shape[0]))
    with nogil, parallel():
        for i in prange(xarr.shape[0], schedule='guided'):
            for j in xrange(yarr.shape[0]):
{in_loop_assigns_par}
                ret[i,j] = {name}({arg_names})
    return ret
"""

__ARG_DEFN_TEMPLATE__ = "cdef double {}"
__ARG_EXTRACT_TEMPLATE__ = """{} = {}[{}]"""
__ARG_EXTRACT_TEMPLATE_2D__ = """{} = {}[{},{}]"""

__ARG_SPEC_TEMPLATE__ = "double {}"

__SETUP_PY__ = """
try:\n
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = [
        Extension(
            '{pyx_file_name}',
            ['{pyx_file_name}.pyx'],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],
            include_dirs=[np.get_include()]
        )
    ],
    include_dirs = [np.get_include()]
)
"""

# TODO
# - don't want to adding to sys.path for every compilation
# - print_ccode doesn't quite print cython code, only _very close_. Eg. if I have 2 in there instead of 2. it will fail.

def to_c(operators, kern, symbols, limits, supports):
    limit_codes = []
    if limits is not None:
        for z, z0 in limits:
            try:
                lim = sp.simplify(sp.limit(kern, z, z0))
            except Exception as ex:
                raise Exception('Failed to compute lim {} -> {} of operators {}'.format(z, z0, operators))
            lim_code = sp.printing.ccode(lim)
            equality_condition = '{} == {}'.format(sp.printing.ccode(z), sp.printing.ccode(z0))
            limit_codes.append(__CONDITION__.format(equality_condition, lim_code))
    support_codes = []
    if supports is not None:
        for condition in supports:
            condition_code = sp.printing.ccode(condition)
            support_codes.append(__CONDITION__.format('not (' + condition_code + ')', 0.0))

    c_rep = sp.printing.ccode(kern)

    return '\n'.join(limit_codes) + '\n' + '\n'.join(support_codes) + '\n' + 'return {}'.format(c_rep)


def to_cython_routine(name, operators, kern, symbols, limits, supports):
    contents = to_c(operators, kern, symbols, limits, supports)
    arg_defn = '\n'.join([__ARG_DEFN_TEMPLATE__.format(a.name) for group in symbols for a in group])
    arg_spec = ', '.join([__ARG_SPEC_TEMPLATE__.format(a.name) for group in symbols for a in group])
    arg_names = ', '.join([sp.printing.ccode(a) for group in symbols for a in group])
    if len(symbols) == 2:
        arg_assignment = ''
    else:
        arg_assignment = '\n'.join([__ARG_EXTRACT_TEMPLATE__.format(a.name, 'fun_args', ix) for ix, a in enumerate(symbols[2])])

    x_assigns = '\n'.join([__ARG_EXTRACT_TEMPLATE_2D__.format(a.name, 'xarr', 'i', ix) for ix, a in enumerate(symbols[0])])
    y_assigns = '\n'.join([__ARG_EXTRACT_TEMPLATE_2D__.format(a.name, 'yarr', 'j', ix) for ix, a in enumerate(symbols[1])])
    in_loop_assigns = x_assigns + '\n' + y_assigns
    defns = arg_defn + '\n\n' + arg_assignment
    return __ROUTINE__.format(name=name,
                              impl=indent(contents),
                              definitions_and_assignments=indent(defns),
                              in_loop_assigns=indent(in_loop_assigns, 3),
                              in_loop_assigns_par=indent(in_loop_assigns, 4),
                              gram_ext=__GRAM_EXT__,
                              par_ext=__PAR_EXT__,
                              arg_names=arg_names,
                              args=arg_spec,
                              comment=operators)


def indent(code, n=1):
    return re.sub(r'^(.*)$', '    '*n + r'\1', code, flags=re.M)


def randomword(length):
    return ''.join(random.choice(string.lowercase) for _ in range(length))


def compile_cython(cython, root_dir_name=None, clean=True):
    root_dir_name = os.path.join(tempfile.gettempdir(), 'code_tmp') if root_dir_name is None else root_dir_name
    if not os.path.exists(root_dir_name):
        os.mkdir(root_dir_name)
    mod_name = randomword(8)
    dir_name = os.path.join(root_dir_name, mod_name)
    os.mkdir(dir_name)

    with tempfile.NamedTemporaryFile(dir=dir_name, delete=False, suffix='.pyx') as f:
        to_import = os.path.splitext(os.path.basename(f.name))[0]
        f.write(cython)
    with open(os.path.join(dir_name, 'setup.py'), 'w') as f:
        f.write(__SETUP_PY__.format(pyx_file_name=to_import))

    __run_setup__([sys.executable, 'setup.py', 'build_ext', '--inplace'], dir_name)
    if dir_name not in sys.path: sys.path.append(dir_name)
    mod = __import__(to_import)
    if clean:
        shutil.rmtree(dir_name)

    return mod


def __check_symbols__(symbols):
    assert len(symbols) in (2,3), "Require at least 3 elements in symbols list"
    assert len(symbols[0]) == 0, "Currently only support 1D"
    assert len(symbols[1]) == 0, "Currently only support 1D"


def __run_setup__(command, cwd):
    e = os.environ.copy()
    e['CC'] = '/usr/local/bin/gcc-5'
    try:
        retoutput = check_output(command, stderr=STDOUT, cwd=cwd, env=e)
    except CalledProcessError as e:
        raise Exception(
            "Error while executing command: %s. Command output is:\n%s" % (
                " ".join(command), e.output.decode()))


def compile_sympy(ops, ops_bar, kern, symbols, parallel=False, limits=None, supports=None, clean=True):
    op_map = {}
    all_combs = [()] + [(o,) for o in ops] + [(o,) for o in ops_bar] + [(o, o_bar) for o in ops for o_bar in ops_bar]

    codes = []
    for comb in all_combs:
        k_op = kern
        for op in comb:
            k_op = op(k_op)
        identifier = randomword(8)
        code = to_cython_routine(identifier, comb, k_op, symbols, limits, supports)
        op_map[comb] = identifier
        codes.append(code)
    final_code = __HEADER__ + '\n\n'.join(codes)
    mod = compile_cython(final_code, clean=clean)

    return SympyModuleOperatorSystem(ops, ops_bar, mod, op_map, parallel=parallel)


class SympyModuleOperatorSystem(object):
    def __init__(self, ops, ops_bar, module, op_map, parallel=False):
        self.__ops__ = ops
        self.__ops_bar__ = ops_bar
        self.__module__ = module
        compilation_utils.infill_op_dict(ops, ops_bar, op_map)
        self.__op_map__ = op_map
        self.__parallel__ = parallel


    @property
    def operators(self):
        return self.__ops__

    @property
    def operators_bar(self):
        return self.__ops_bar__

    def __getitem__(self, item):
        if item not in self.__op_map__:
            raise Exception("Don't have any operator corresponding to {}".format(item))
        name = self.__op_map__[item] + (__GRAM_EXT__ if not self.__parallel__ else __PAR_EXT__)
        return getattr(self.__module__, name)