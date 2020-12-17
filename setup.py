#! /usr/bin/env python

"""
Installer script for the ``imc`` package.

Install with ``pip install .``.
"""

from setuptools import setup, find_packages
from pathlib import Path


def parse_requirements(req_file):
    """Parse requirements.txt files."""
    reqs = open(req_file).read().strip().split("\n")
    reqs = [r for r in reqs if not r.startswith("#")]
    return [r for r in reqs if ("#egg=" not in r) and (r != "")]


# Requirements
reqs_dir = Path("requirements")
reqs = dict()
for f in reqs_dir.iterdir():
    pkgs = parse_requirements(f)
    if "." in f.stem:
        k = f.stem.split(".")[-1]
        reqs[k] = pkgs
    else:
        reqs["base"] = pkgs

# Description
long_description = open("README.md").read()


# setup
setup(
    name="imc",
    packages=find_packages(),
    use_scm_version={
        "write_to": "imc/_version.py",
        "write_to_template": '__version__ = "{version}"\n',
    },
    entry_points={
        "console_scripts": [
            "imc = imc.cli:main",
        ]
    },
    description="A framework for IMC data analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 3 - Alpha",
        "Typing :: Typed",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords=",".join(
        [
            "computational biology",
            "bioinformatics",
            "imaging mass cytometry",
            "imaging",
            "mass cytometry",
            "mass spectrometry",
        ]
    ),
    url="https://github.com/elementolab/imc",
    project_urls={
        "Bug Tracker": "https://github.com/elementolab/imc/issues",
        # "Documentation": "https://imc.readthedocs.io",
        "Source Code": "https://github.com/elementolab/imc",
    },
    author=u"Andre Rendeiro",
    author_email="andre.rendeiro@pm.me",
    license="GPL3",
    setup_requires=["setuptools_scm"],
    install_requires=reqs["base"],
    tests_require=reqs["dev"],
    extras_require={k: v for k, v in reqs.items() if k not in ["base", "dev"]},
    # package_data={"imc": ["config/*.yaml", "templates/*.html", "_models/*"]},
    # data_files=[("requirements", reqs.values())],
)
