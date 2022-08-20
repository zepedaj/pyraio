Building PyRAIO for new architectures
--------------------------------------

|PyRAIO| uses |Cython| to build a |libaio| wrapper.

Wrappers for alternate architectures can be easily built with the help of |autopxd2|. This Python tool can be installed via pip (see the `python-autopxd2 github page <https://github.com/gabrieldemarmiesse/python-autopxd2>`__ if you want more info):

.. code-block:: console

    pip install autopxd2

You will also need to install the |libaio| Linux library with header files, if not already installed:

.. code-block:: console

   # Ubuntu 20.04
   sudo apt install -y libaio1 libaio-dev

You then generate the required |Cython| |pxd| file using |autopxd2|:

.. code-block:: console

   autopxd2 path/to/libaio.h clibio_<my_arch_name>.h

The produced header file will rely on two |C| structures that are not defined in that header file. These structures can be defined as opaque structures by inserting the following lines at the top of the ``cdef extern`` context within the generated |pxd| file:

.. code-block:: cython

    cdef extern from "libaio.h": # This line is already in the generated pxd file

        # Insert starting here
        ctypedef struct io_context:
            pass

        ctypedef struct sigset_t:
            pass

Loading this file will require modifying the ``pyraio.pyx`` source file to select the correct wrapper for the architecture.

.. note:: Consider submitting a github PR with the |pxd| file for the new architecture and the modifications to the ``pyraio.pyx`` file.
