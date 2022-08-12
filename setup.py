# !/usr/bin/env python

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="pyraio",
    ext_modules=cythonize(
        [
            Extension(
                "pyraio",
                ["pyraio.pyx"],
                libraries=["aio"],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
        ],
        annotate=True,
        language_level="3",
    ),
    version="0.1.0",
    description="",
    author="Joaquin Zepeda",
    include_dirs=[np.get_include()],
)
