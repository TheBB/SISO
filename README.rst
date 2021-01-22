====
SISO
====

.. image:: https://badge.fury.io/py/SISO.svg
   :target: https://badge.fury.io/py/SISO

.. image:: https://travis-ci.org/TheBB/SISO.svg?branch=master
   :target: https://travis-ci.org/TheBB/SISO


SISO is a tool for converting between various data formats used for
storing simulation results.

Supported readers:

- IFEM_ HDF5 files (.hdf5)
- LRSpline_ geometries (.lr)
- GoTools_ (B-Spline) geometries (.g2)
- SIMRA_ result files (.res)
- WRF_ result files (NetCDF4)

Supported writers:

- PVD/VTU/VTK: for use with Paraview_
- VTF: for use with GLView_


Installation
------------

It is recommended to install with PIP::

  pip install --user SISO


To install from source::

  pip install --user .


SISO requires Python 3.  It is possible that, on your system, *pip*
refers to Python 2.  In this case, do::

  pip3 install --user SISO


or::

  pip3 install --user .


SISO requires the numpy_, H5Py_, VTK_ and LRSplines_ libraries, all of
which contain compiled components.  In ideal circumstances, they
should be installed automatically from PyPi with the above commands,
but circumstances are not always ideal.  For example, VTK is, as of
the time of writing, not available for Python 3.8 on PyPi.  If
dependencies fail to install, please consult the relevant
documentation of the respective libraries.

VTF support is not available out-of-the-box because it depends on
proprietary libaries.  If these libraries are available on your
system, you may install VTFWriter_ manually.  If it is present, SISO
should enable VTF support automatically.

Upon successful installation, a *siso* executable should be installed
in ``~/.local/bin``, or the binary path of the current Python
environment.  To run it, ensure that this directory is in your
``PATH``.


Usage
-----

Basic usage is::

  siso INFILE.hdf5 [OUTFILE]


For help with relevant command-line-options please consult::

  siso --help


If the output filename is specified, the format will be determined
from its extension, unless specifically overriden with ``-f`` or
``--fmt``::

  siso -f vtu INFILE.hdf5


Some options are only available for certain input or output formats,
and for some options some specific values may only be available for
certain input or output formats.  Further, depending on circumstances,
some options may be determined to be incompatible with each other.  In
all these cases you should be either warned or provided with an error
message.  If you feel you get an error in a case where that shouldn't
happen, please open an issue, including the information from running
with ``--debug``.



Logging options
^^^^^^^^^^^^^^^

The verbosity of standard output can be changed with ``--debug`` (use
for bug reports), ``--info`` (default), ``--user``, ``--warning`` and
``--error``.

SISO will try to use rich formatted output by default.  Switch this
off with ``--no-rich``.



General options
^^^^^^^^^^^^^^^

To increase resolution beyond linear interpolation between element
vertices, use the ``-n`` or ``--nvis`` option.  A value of e.g. 3
indicates that each element should be subdivided into three per axis,
creating a 9-fold increase in data amount for 2D results.

To include only certain fields use the ``--filter`` or ``-l`` option.
Issue this option multiple times, or give a comma-separated list as
argument.

Use ``-f`` or ``--fmt`` to override the output format, which is by
default derived from the output filename extension.  This may be
necessary for formats which write to directories.

Use ``--last`` to only write the final timestep.

Use ``--mode`` to specify output mode, depending on what the writer is
capable of.  All current writers are capable of switching between
``binary`` (default) and ``ascii`` mode.



WRF options
^^^^^^^^^^^

Output is given in projected coordinates by default.  If ``--global``
is given, they will be converted to true physical coordinates, with
the origin at the center of the Earth, the x-axis pointing toward the
prime meridian and the z-axis pointing toward the north pole.

By default, the output is a 3D mesh including all volumetric fields.
If ``--planar`` is given, it will be a 2D mesh including only planar
fields, as well as the surface slice of volumetric fields.  If
``--extrude`` is given, it will be a 3D mesh including all volumetric
fields, as well as planar fields interpreted as constants in the
vertical direction.

If ``--periodic`` is given, SISO will interpret the mesh as a global
grid and attempt to tie it together in the longitudinal axis and at
the poles.  This only works together with ``--global``.



IFEM options
^^^^^^^^^^^^

The output may be restricted to certain bases with ``--basis`` or
``-b``. Use this option multiple times or give a comma-separated list
as argument.

The basis for the geometry can be chosen with ``--geometry``.



SIMRA options
^^^^^^^^^^^^^

The endianness of the input can be specified with ``--endianness``,
with valid arguments being ``native`` (default), ``little`` and
``big``.


.. _IFEM: https://github.com/OPM/IFEM
.. _LRSpline: https://github.com/VikingScientist/LRSplines
.. _GoTools: https://github.com/SINTEF-Geometry/GoTools
.. _SIMRA: https://www.sintef.no/en/digital/applied-mathematics/simulation/computational-fluid-dynamics1/
.. _WRF: https://www.mmm.ucar.edu/weather-research-and-forecasting-model
.. _Paraview: https://www.paraview.org/
.. _GLView: https://ceetron.com/ceetron-glview-inova/
.. _numpy: https://numpy.org/
.. _H5Py: https://www.h5py.org/
.. _VTK: https://vtk.org/
.. _LRSplines: https://github.com/TheBB/lrsplines-python
.. _VTFWriter: https://github.com/TheBB/vtfwriter
