==========
IFEM-to-VT
==========

.. image:: https://badge.fury.io/py/IFEM-to-VT.svg
   :target: https://badge.fury.io/py/IFEM-to-VT

.. image:: https://travis-ci.org/TheBB/IFEM-to-VT.svg?branch=master
   :target: https://travis-ci.org/TheBB/IFEM-to-VT


IFEM-to-VT is a tool for converting IFEM_ simulation results to other
formats more suitable for visualization:

- VTU/VTK: for use with Paraview_
- VTF: for use with GLView_

So named because all supported formats to date start with "VT".


Installation
------------

It is recommended to install with PIP::

  pip install --user .


IFEM-to-VT requires Python 3.  It is possible that, on your system,
*pip* refers to Python 2.  In this case, do::

  pip3 install --user .


IFEM-to-VT requires the numpy_, H5Py_, VTK_ and LRSplines_ libraries, all
of which contain compiled components.  In ideal circumstances, they
should be installed automatically from PyPi with the above command,
but circumstances are not always ideal.  For example, VTK is, as of
the time of writing, not available for Python 3.8 on PyPi.  If
dependencies fail to install, please consult the relevant
documentation of the respective libraries.

VTF support is not available out-of-the-box because it depends on
proprietary libaries.  If these libraries are available on your
system, you may install VTFWriter_ manually.  If it is present,
IFEM-to-VT should enable VTF support automatically.

Upon successful installation, an *ifem-to-vt* executable should be
installed in ``~/.local/bin``, or the binary path of the current
Python environment.  To run it, ensure that this directory is in your
``PATH``.


Usage
-----

Basic usage is::

  ifem-to-vt INFILE.hdf5 [OUTFILE]


For help with relevant command-line-options please consult::

  ifem-to-vt --help


If the output filename is specified, the format will be determined
from its extension, unless specifically overriden with ``-f`` or
``--fmt``::

  ifem-to-vt -f vtu INFILE.hdf5


You can restrict the output to certain bases by using the ``-b`` or
``--basis`` option. It can be given multiple times, for example::

  ifem-to-vt --basis NavierStokes-1 --basis AdvectionDiffusion-1 INFILE.hdf5


By default, the first basis in the file will be used for the
geometry.  To override this, use the ``-g`` or ``--geometry`` option::

  ifem-to-vt --geometry NavierStokes-1 INFILE.hdf5


To increase resolution beyond linear interpolation between element
vertices, use the ``-n`` or ``--nvis`` option.  A value of e.g. 3
indicates that each element should be subdivided into three per axis,
creating a 9-fold increase in data amount for 2D results::

  ifem-to-vt --nvis 3 INFILE.hdf5


Both VTF, VTK and VTU formats support ASCII and binary modes.  By
default, IFEM-to-VT writes binary files.  To override this, use the
``-m`` or ``--mode`` option, with value ``ascii``, ``binary`` or
``appended`` (the latter only supported for VTU).

The verbosity of the output can be controlled with the ``-v`` or
``--verbosity`` option, with values ``debug``, ``info`` (default),
``warning``, ``error`` and ``critical``.  For submitting bug reports,
please attach the log with ``-v debug``, and if possible a sample HDF5
file which reproduces the error.


.. _IFEM: https://github.com/OPM/IFEM
.. _Paraview: https://www.paraview.org/
.. _GLView: https://ceetron.com/ceetron-glview-inova/
.. _numpy: https://numpy.org/
.. _H5Py: https://www.h5py.org/
.. _VTK: https://vtk.org/
.. _LRSplines: https://github.com/TheBB/lrsplines-python
.. _VTFWriter: https://github.com/TheBB/vtfwriter
