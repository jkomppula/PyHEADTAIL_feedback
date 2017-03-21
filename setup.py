from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize

#  run the command python setup.py build_ext --inplace

cy_ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
cy_ext = [
        Extension("cython_functions",
                 ["cython_functions.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"]
                 )
        ]

setup(
    name='PyHEADTAIL_feedback',
    ext_modules=cythonize(cy_ext, **cy_ext_options),
    )