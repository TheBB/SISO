#!/usr/bin/env python

from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name='IFEM-to-VT',
    version='0.0.1',
    description='Converts IFEM result files to VT* formats.',
    maintainer='Eivind Fonn',
    maintainer_email='eivind.fonn@sintef.no',
    ext_modules=cythonize(Extension(
        'vtfwriter',
        ['vtfwriter.pyx'],
        libraries=['VTFExpressAPI'],
        library_dirs=['/usr/local/lib', '/usr/local/lib64']
    )),
    packages=['ifem_to_vt'],
    install_requires=['numpy', 'Splipy', 'h5py'],
    entry_points={
        'console_scripts': [
            'ifem-to-vt=ifem_to_vt.__main__:convert',
        ],
    },
)
