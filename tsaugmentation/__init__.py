__version__ = "0.5.38"

from tsaugmentation import preprocessing
from tsaugmentation import transformations
from tsaugmentation import visualization

# Only print in interactive mode
import __main__ as main
if not hasattr(main, '__file__'):
    print("""Importing the tsaugmentation module. L. Roque. 
    Method to Test the Robustness of Hierarchical Time Series Forecasting Algorithms.\n""")