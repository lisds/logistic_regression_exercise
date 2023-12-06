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

def logistic_regression_cost_function(int_slo, x, y):

    """
    This function calculates the cost function for logistic regression.
    """

    intercept, slope = int_slo
    
    # Make predictions on the log odds (straight line) scale
    predicted_log_odds = intercept + slope * x

    # Convert these predictions to probabilities.
    predicted_prob_of_1 = inverse_logit(predicted_log_odds)
    
    # Calculate predicted probabilities of the actual scores (THIS IS THE MOST IMPORTANT STEP TO UNDERSTAND!)
    predicted_prob_of_actual_scores = y * predicted_prob_of_1 + (1 - y) * (1 - predicted_prob_of_1)

    # Multiply the predicted probabiity of the actual score
    likelihood = np.prod(predicted_prob_of_actual_scores)
    
    # Ask minimize to find maximum by adding minus sign.
    return -likelihood