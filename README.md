# Logistic regression exercise task

This is an template repo for the logistic regression exercise task.
In this exercise you will perform some simple logistic regression analysis using
the new workflow (e.g. using the command line, scripts, git and github). You should:

- Fork this repository
- use `git clone` to create a local copy of your fork (e.g. on your laptop)
- Download the full CKD dataset into your repository, from here: https://github.com/matthew-brett/cfd2020/blob/master/data/ckd_full.csv

Your task is then to fit the following logistic regression model `Class ~ Hemoglobin`.
E.g. predicting whether or not a patient has CKD from their hemoglobin score.

To do this you should modify the following files:

- COST FUNCTION MODULE: (`cost_function_utils.py`) this .py should contain an
  inverse logit transformation function, and the logistic regression cost function.
  DO NOT COPY/PASTE the functions from the textbook page, try to
  re-write them line-by-line yourselves so you understand what each line is doing.
  These functions can then can imported by other scripts in the repository, e.g.
  if this file is called "cost_function_utils.py" then you can import the
  function into other scripts using `import cost_function_utils`. You should 
  import the cost function into your data analysis script (`data_analysis.py`)

- DATA ANALYSIS SCRIPT: this file (`data_analysis.py`) should do several things:
  
    - First, this script should plot `Class` as a function of `Hemoglobin`.
       
    - Then the script should fit your model (`Class ~ Hemoglobin`) using
        `minimize` (by importing your cost function `cost_function_utils.py`).
      
    - You should then fit the same model using `statsmodels`.
      
    - You should write some tests to check that `minimize` and `statsmodels`
      are producing similar parameter estimates (HINT: you may want to investigate
      `np.isclose()` and `assert` - ask us for help with this also).
      
    - You should then create a new plot, showing `Class ~ Hemoglobin`, but 
      also showing the predictions from your logistic regression model.

The file `showcase_notebook.Rmd` should be used **only** to run your 
`data_analysis.py` script (e.g. to fit the models and show the plots).

You should then submit your work by sending a pull request to merge the
changes on your branch back to the repository you forked from, e.g.: 
https://github.com/lisds/logistic_regression_exercise
