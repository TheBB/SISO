#!/usr/bin/env python

from pathlib import Path
from setuptools import setup
from distutils.extension import Extension

with open(Path(__file__).parent / 'README.rst') as f:
    desc = f.read()

setup(
    name='IFEM-to-VT',
    version='2.0.2',
    description='Convert between different mesh data formats',
    long_description_content_type='text/x-rst',
    long_description=desc,
    maintainer='Eivind Fonn',
    maintainer_email='eivind.fonn@sintef.no',
    packages=['ifem_to_vt', 'ifem_to_vt.writer', 'ifem_to_vt.reader'],
    install_requires=[
        'click',
        'dataclasses',
        'numpy',
        'Splipy>=1.4',
        'lrsplines>=1.5',
        'h5py',
        'vtk',
        'netcdf4',
        'nptyping',
        'singledispatchmethod',
        'treelog',
    ],
    extras_require={
        'VTF': ['vtfwriter'],
        'testing': ['pytest'],
        'deploy': ['twine', 'cibuildwheel==1.1.0'],
    },
    entry_points={
        'console_scripts': [
            'ifem-to-vt=ifem_to_vt.__main__:convert',
        ],
    },
)
