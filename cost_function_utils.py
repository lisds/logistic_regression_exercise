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
    """ Reverse logit transformation on array `v`"""
    return np.exp(v) / (1 + np.exp(v)) # Reverse the log operation and odds operation.

def logistic_regression_cost_function(int_slo, x_val, y_val):

    # unpack variables
    intercept, slope = int_slo

    # predicted vals for log-odds straight line
    predicted_logg_odds = intercept + slope * x_val

    # converting straight line predictions to a sigmoid probability curve
    predicted_probs = inverse_logit(predicted_logg_odds)

    # calc prediction error
    sigmoid_err = y_val - predicted_probs

    return np.sum(sigmoid_err**2)