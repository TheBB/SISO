#!/usr/bin/env python

from setuptools import setup
from distutils.extension import Extension

setup(
    name='IFEM-to-VT',
    version='1.0.0',
    description='Converts IFEM result files to VT* formats.',
    maintainer='Eivind Fonn',
    maintainer_email='eivind.fonn@sintef.no',
    packages=['ifem_to_vt', 'ifem_to_vt.writer', 'ifem_to_vt.reader'],
    install_requires=['click', 'numpy', 'Splipy>=1.4', 'lrsplines>=1.5', 'h5py', 'vtk'],
    extras_require={
        'VTF': ['vtfwriter'],
    },
    entry_points={
        'console_scripts': [
            'ifem-to-vt=ifem_to_vt.__main__:convert',
        ],
    },
)
