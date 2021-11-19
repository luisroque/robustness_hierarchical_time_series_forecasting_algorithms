from setuptools import setup, find_packages
from os.path import dirname, join, realpath

VERSION = '0.0.1'
DESCRIPTION = 'Robust time series augmentation for forecasting algorithms'
LONG_DESCRIPTION = 'A package that allows to augment your time series dataset ' \
                   'to test the robustness of forecasting algorithms'

PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")
with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

# Setting up
setup(
    name="tsaugmentation",
    version=VERSION,
    author="Luis Roque",
    author_email="<roque0luis@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=install_reqs,
    keywords=['python', 'time series', 'hierarchical', 'forecasting', 'augmentation', 'machine learning'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
