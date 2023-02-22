![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tsaugmentation)
[![license](https://img.shields.io/badge/License-BSD%203-brightgreen)](https://github.com/luisroque/robustness_hierarchical_time_series_forecasting_algorithms/blob/main/LICENSE)
![PyPI](https://img.shields.io/pypi/v/tsaugmentation)
[![Downloads](https://pepy.tech/badge/tsaugmentation)](https://pepy.tech/project/tsaugmentation)

# RHiOTS: A Framework for Evaluating Hierarchical Time Series

This repository provides the implementation of RHiOTS, a novel methodology for evaluating the robustness of HTS (hierarchical time series) forecasting algorithms on real-world datasets. It includes a comprehensive framework that extends beyond the traditional evaluation of predictive performance, enabling a more reliable method for selecting the appropriate HTSF algorithm for a given problem than existing benchmark-based approaches. Additionally, the repository provides a set of parameterizable transformations to simulate changes in the data distribution.

You can install tsagumentation as a python package
```python
pip install tsaugmentation
```

## Functionality


The main functionality of this repository includes:

* **RHiOTS Methodology:** The implementation of the RHiOTS methodology for evaluating the robustness of HTS forecasting algorithms on real-world datasets.
* **Comprehensive Framework:** A comprehensive framework that extends beyond the traditional evaluation of predictive performance, including methods for evaluating forecast accuracy, forecast stability, and forecast bias.
* **Algorithm Selection:** A more reliable method for selecting the appropriate HTSF algorithm for a given problem than existing benchmark-based approaches.
* **Parameterizable Transformations:** A set of parameterizable transformations to simulate changes in the data distribution.


## Getting started
The code below creates new versions of the prison dataset by applying time series augmentation transformations.

```python
import tsaugmentation as tsag

# Creates a transformation class
data = tsag.transformations.CreateTransformedVersions(
    dataset_name='prison'
)

# Creates new versions of the prison dataset
data.create_new_version_single_transf()
```

### Contributing
We welcome contributions to this repository. If you find a bug, or if you have an idea for a new feature, please open an issue or submit a pull request.

### License
This repository is licensed under the BSD 3-Clause License. See the LICENSE file for more information.