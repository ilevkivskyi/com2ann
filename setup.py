#!/usr/bin/env python

from setuptools import setup

version = '0.1.0'
description = 'Tool to translate type comments to annotations.'
long_description = '''
com2ann
=======

Tool for translation type comments to type annotations in Python.

The tool requires Python 3.8 to run. But the supported target code version
is Python 3.4+ (can be specified with ``--python-minor-version``).

Currently, the tool translates function and assignment type comments to
type annotations.

The philosophy of of the tool is too minimally invasive, and preserve original
formatting as much as possible. This is why the tool doesn't use un-parse.
'''.lstrip()

classifiers = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3.8',
    'Topic :: Software Development',
]

setup(
    name='com2ann',
    version=version,
    description=description,
    long_description=long_description,
    author='Ivan Levkivskyi',
    author_email='levkivskyi@gmail.com',
    url='https://github.com/ilevkivskyi/com2ann',
    license='MIT',
    keywords='typing function annotations type hints '
             'type comments variable annotations',
    package_dir={'': 'src'},
    py_modules=['com2ann'],
    entry_points={'console_scripts': ['com2ann=com2ann:main']},
    classifiers=classifiers,
)
