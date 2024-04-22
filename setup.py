# !/usr/bin/env python

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

setup(
    # name="pyraio",
    # install_requires=["jztools", "cython"],
    ext_modules=cythonize(
        [
            Extension(
                f"pyraio.{_mdl}",
                [f"pyraio/{_mdl}.pyx"],
                libraries=["aio", "uring"],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
            for _mdl in ["batch_reader", "read_input_iter", "event_managers"]
        ],
        annotate=True,
        language_level="3",
    ),
    zip_safe=False,
    # version="0.1.0",
    # author="Joaquin Zepeda",
    include_dirs=[np.get_include()],
    # description="Pyraio: SSD random io",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # url="https://github.com/zepedaj/pyraio",
)
