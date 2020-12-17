#! /usr/bin/env python

"""
Installer script for the ``imc`` library and the ``imcpipeline`` pipeline.

Install with ``pip install .``.
"""

from setuptools import setup, find_packages


def parse_requirements(req_file):
    """Parse requirements.txt files."""
    reqs = open(req_file).read().strip().split("\n")
    reqs = [r for r in reqs if not r.startswith("#")]
    return [r for r in reqs if "#egg=" not in r]


REQUIREMENTS_FILE = "requirements.txt"
DEV_REQUIREMENTS_FILE = "requirements.dev.txt"
README_FILE = "README.md"

# Requirements
requirements = parse_requirements(REQUIREMENTS_FILE)
requirements_dev = parse_requirements(DEV_REQUIREMENTS_FILE)
requirements_extra = {
    "stardist": ["stardist==0.6.0,<1.0.0"],
    "deepcell": ["DeepCell==0.8.3,<1.0.0"],
    "cellpose": ["cellpose>=0.1.0.1,<1.0.0"],
}

# Description
long_description = open(README_FILE).read()


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
    description="A pipeline and utils for IMC data analysis.",
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
    install_requires=requirements,
    tests_require=requirements_dev,
    extras_require={"dev": requirements_dev, **requirements_extra},
    # package_data={"imc": ["config/*.yaml", "templates/*.html", "_models/*"]},
    data_files=[
        REQUIREMENTS_FILE,
        # "requirements/requirements.test.txt",
    ],
)
