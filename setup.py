# !/usr/bin/env python

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="pyraio",
    ext_modules=cythonize(
        [
            Extension(
                f"pyraio.{_mdl}",
                [f"pyraio/{_mdl}.pyx"],
                libraries=["aio", "uring"],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
            for _mdl in [
                "reader",
                "batch_reader",
                "batch_reader_uring",
                "util",
            ]
        ],
        annotate=True,
        language_level="3",
    ),
    version="0.1.0",
    description="",
    author="Joaquin Zepeda",
    include_dirs=[np.get_include()],
)
