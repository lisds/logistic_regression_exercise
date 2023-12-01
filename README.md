# Logistic regression exercise task

This is an template repo for the logistic regression exercise task.
In this exercise you will perform some simple logistic regression analysis using
the new workflow (e.g. using the command line, scripts, git and github). You should:

- Fork this repository
- use `git clone` to create a local copy of your fork (e.g. on your laptop)
- Download the full CKD dataset into your repository, from here: https://github.com/matthew-brett/cfd2020/blob/master/data/ckd_full.csv

Your task is then to fit the following logistic regression model `Class ~ Hemoglobin`.
E.g. predicting whether or not a patient has CKD from their hemoglobin score.

To do this you should write the following set of scripts:

- COST FUNCTION SCRIPT: create a .py file containing the logit transformation
 function, the inverse logit transformation function, and the logistic regression
 cost function. DO NOT COPY/PASTE the functions from the textbook page, try to
 re-write them line-by-line yourselves so you understand what each line is doing.
 These functions can then can imported by other scripts in the repository, e.g.
 if this file is called "cost_function_utils.py" then you can import the
function into other scripts using `import cost_function_utils`

- DATA ANALYSIS SCRIPT: this .py file should do several things:
        - First, this script should plot `Class` as a function of `Hemoglobin`. 
        - Then the script should fit your model (`Class ~ Hemoglobin`) using
         `minimize` (by importing your cost function from COST FUNCTION SCRIPT). 
        - You should then fit the same model using statsmodels.
        - You should write some tests to check that `minimize` and `statsmodels`
          are producing similar parameter estimates (HINT: you may want to investigate
         `np.isclose()` and `assert` - ask us for help with this also).
        - You should then create a new plot, showing `Class ~ Hemoglobin`, but 
        also showing the predictions from your logistic regression model.

Once the scripts are written, the full analysis steps are:

- run the data cleaning script (this will save a cleaned .csv file)
- run the data analysis script (this will import the cost function from
  cost_function_utils.py and will generate the graphs/perform the analysis)

You could also set up the data analysis script to call the data cleaning script
directly (see here: https://stackoverflow.com/questions/7974849/how-can-i-make-one-python-file-run-another).

The `showcase_notebook.ipynb` is to run the data analysis script, for
presentation etc. (Normally we would prefer .Rmd files, but this notebook
is only a few lines long, and is only being used to run one other script, so it
is OK in this case).
