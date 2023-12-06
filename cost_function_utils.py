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
    # your code here
    return np.exp(v) / (1 + np.exp(v))

def logistic_regression_cost_function(intercept_and_slope, x_vals, y_vals):
    #unpacking into values 
    intercept, slope = intercept_and_slope
    #log odds into a straight line
    predicted_log_odds = intercept + slope * x_vals
    #Converting Predictions to probabilities
    predicted_prob_of_1 = inverse_logit(predicted_log_odds)
    # Calculating predicted probabilities 
    predicted_prob_of_actual_scores = y_vals * predicted_prob_of_1 + (1 - y_vals) * (1 - predicted_prob_of_1)
    # Multiplying predicted probably by actual probability
    log_likelihood = np.sum(np.log(predicted_prob_of_actual_scores))
    #Mean of the root of squared errors
    return -log_likelihood