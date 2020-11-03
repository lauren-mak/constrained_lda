#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools


setuptools.setup(
    name='constrained-lda',
    version='0.1.0',
    description="",
    author="Lauren Mak",
    author_email='',
    url='',
    packages=setuptools.find_packages(),
    package_dir={'constrained-lda': 'constrained-lda'},
    install_requires=[
        'click',
        'pandas',
        'scipy',
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'constrained-lda=constrained-lda.cli:main'
        ]
    },
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)
