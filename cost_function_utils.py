# this module contains functions related to the manual implementation of the 
# logistic regression cost function
import numpy as np

# this is just to demonstrate how functions from this file can be used in 
# "showcase_notebook.Rmd" - you can delete this function 
def junk_function():
    """This is just a junk function to show how you can import functions from this
    file into your `showcase_notebook`"""
    print('The function that produced this printout was imported from `cost_function_utils.py`')

def inverse_logit(log_odds):
    odds = np.exp(log_odds)
    p = odds/(1 + odds)
    return p

#I have chosen sum of squared errors as it feels more intuitive and I don't fully understand the benefits of using likelihood as cost
def logistic_regression_cost_function(intercept_slope, x_values, y_values):
    intercept, slope = intercept_slope
    predicted_log_odds = slope * x_values + intercept
    sigmoid_values = inverse_logit(predicted_log_odds)
    sigmoid_error = y_values - sigmoid_values
    return np.sqrt(np.mean(sigmoid_error ** 2))
