from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Robust time series augmentation for forecasting algorithms'
LONG_DESCRIPTION = 'A package that allows to augment your time series dataset ' \
                   'to test the robustness of forecasting algorithms'

# Setting up
setup(
    name="rts",
    version=VERSION,
    author="Luis Roque",
    author_email="<roque0luis@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    keywords=['python', 'time series', 'hierarchical', 'forecasting', 'augmentation', 'machine learning'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Researchers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)