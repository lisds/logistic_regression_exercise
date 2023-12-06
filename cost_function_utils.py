# this module contains functions related to the manual implementation of the 
# logistic regression cost function
import numpy as np

# this is just to demonstrate how functions from this file can be used in 
# "showcase_notebook.Rmd" - you can delete this function 
def junk_function():
    """This is just a junk function to show how you can import functions from this
    file into your `showcase_notebook`"""
    print('The function that produced this printout was imported from `cost_function_utils.py`')

def inverse_logit(v):
    odds = np.exp(v)
    return odds / (odds + 1)

def logistic_regression_cost_function(intercept_and_slope, x_values, y_values):
    intercept, slope = intercept_and_slope
    predicted_log_odds = intercept + slope * x_values
    predicted_probabilities = inverse_logit(predicted_log_odds)
    sigmoid_error = y_values - predicted_probabilities
    return np.sum(sigmoid_error ** 2)
