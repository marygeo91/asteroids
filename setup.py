#!/bin/env/ python

from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
        ext_modules=cythonize("sj.pyx"),
        include_dirs=[numpy.get_include()]
        )

