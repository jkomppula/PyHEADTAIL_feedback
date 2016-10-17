from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "Cython dot",
    ext_modules = cythonize('cython_dot.pyx'),  # accepts a glob pattern
)