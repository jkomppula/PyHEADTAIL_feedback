from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize

#  run the command python setup.py build_ext --inplace

cy_ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
cy_ext = [
        Extension("processors.cython_hacks",
                 ["processors/cython_hacks.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"]
                 ),

        Extension("signal_tools.cython_hacks",
                 ["signal_tools/cython_hacks.pyx"],
                 include_dirs=[np.get_include()], library_dirs=[], libraries=["m"]
                 )
        ]

setup(
    name='PyHEADTAIL_feedback',
    ext_modules=cythonize(cy_ext, **cy_ext_options),
    )
