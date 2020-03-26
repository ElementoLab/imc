#! /usr/bin/env python

import sys


def parse_requirements(req_file):
    requirements = open(req_file).read().strip().split("\n")
    requirements = [r for r in requirements if not r.startswith("#")]
    return [r for r in requirements if "#egg=" not in r]


# take care of extra required modules depending on Python version
extra = {}
try:
    from setuptools import setup, find_packages

    if sys.version_info < (2, 7):
        extra["install_requires"] = ["argparse"]
    if sys.version_info >= (3,):
        extra["use_2to3"] = True
except ImportError:
    from distutils.core import setup

    if sys.version_info < (2, 7):
        extra["dependencies"] = ["argparse"]

# Requirements
requirements = parse_requirements(
    "requirements.txt")
# requirements_test = parse_requirements(
#     "requirements/requirements.test.txt")
# requirements_docs = parse_requirements(
#     "requirements/requirements.docs.txt")

long_description = open("README.md").read()


# setup
setup(
    name="imc-pipeline",
    packages=find_packages(),
    use_scm_version={
        'write_to': 'src/_version.py',
        'write_to_template': '__version__ = "{version}"\n'
    },
    entry_points={
        "console_scripts": [
            "imc-pipeline = src.pipeline:main"]
    },
    description="A pipeline and utils for IMC data analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: "
        "GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="bioinformatics, sequencing, ngs, ngs analysis, "
             "ATAC-Seq, ChIP-seq, RNA-seq, project management",
    url="https://github.com/elementolab/hyperion-cytof",
    project_urls={
        "Bug Tracker": "https://github.com/elementolab/hyperion-cytof/issues",
        # "Documentation": "https://imc-pipeline.readthedocs.io",
        "Source Code": "https://github.com/elementolab/hyperion-cytof",
    },
    author=u"Andre Rendeiro",
    author_email="andre.rendeiro@pm.me",
    license="GPL3",
    setup_requires=['setuptools_scm'],
    install_requires=requirements,
    # tests_require=requirements_test,
    # extras_require={
    #     "testing": requirements_test,
    #     "docs": requirements_docs},
    # package_data={"imc-pipeline": ["config/*.yaml", "templates/*.html", "models/*"]},
    data_files=[
        "requirements/requirements.txt",
        "requirements/requirements.test.txt",
    ],
    **extra
)
