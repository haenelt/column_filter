# -*- coding: utf-8 -*-

import setuptools


INSTALL_REQUIREMENTS = ['numpy==1.23.5',
                        'pandas==1.5.2',
                        'nibabel==4.0.2',
                        'gdown==4.6.0',
                        'joblib==1.2.0',
                        'surfdist==0.15.5',
                        'scipy==1.9.3',
                        'tqdm==4.64.1',
                        'pyarrow==10.0.1',
                        ]

CLASSIFIERS = ["Programming Language :: Python :: 3.8",
               "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
               "Operating System :: OS Independent",
               "Development Status :: 3 - Alpha",
               "Intended Audience :: Science/Research",
               "Topic :: Scientific/Engineering",
               ]

with open("VERSION", "r", encoding="utf8") as fh:
    VERSION = fh.read().strip()

with open("README.md", "r", encoding="utf8") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="column_filter",
    version=VERSION,
    author="Daniel Haenelt",
    author_email="daniel.haenelt@gmail.com",
    description="Filter cortical columns on a surface mesh",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/haenelt/column_filter",
    license='GPL v3',
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIREMENTS,
    classifiers=CLASSIFIERS,
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
    )
