from setuptools import setup, find_packages
from os.path import dirname, join, realpath
import re

DESCRIPTION = 'Robust time series tsaugmentation for forecasting algorithms'
LONG_DESCRIPTION = 'A package that allows to augment your time series dataset ' \
                   'to test the robustness of forecasting algorithms'

PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")
with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()


def get_version():
    VERSIONFILE = join("tsaugmentation", "__init__.py")
    lines = open(VERSIONFILE).readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError(f"Unable to find version in {VERSIONFILE}.")

# Setting up
setup(
    name="tsaugmentation",
    version=get_version(),
    author="Luis Roque",
    author_email="<roque0luis@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=install_reqs,
    keywords=['python', 'time series', 'hierarchical', 'forecasting', 'tsaugmentation', 'machine learning'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
