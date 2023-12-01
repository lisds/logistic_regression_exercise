# this script performs logistic regression analysis of the CKD data
import cost_function_utils
import pandas as pd
import numpy as np
from IPython.display import display

# this just shows how to use your `cost_function_utils` module with the current
# script
cost_function_utils.junk_function()

# generate some toy data
data = pd.DataFrame({'A': np.random.normal(100, 10, 100),
                    'B': np.random.normal(10, 1, 100)})

# show the data
display(data)

# create a plot
data.plot(kind='scatter', x='A', y='B')