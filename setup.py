from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='bayesian_pdes',
    version='',
    packages=['bayesian_pdes'],
    url='',
    license='',
    author='benorn',
    author_email='',
    description='',
    requires=['numpy'],
    ext_modules=cythonize('bayesian_pdes/pairwise.pyx'),
    include_dirs=[numpy.get_include()]
)
