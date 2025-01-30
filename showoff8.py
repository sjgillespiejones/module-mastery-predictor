import numpy as np

import utils.expectedValue
from utils import filteredModuleData, gridSearch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

X, y = filteredModuleData.getPassedTestModelData()
params = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}

logistic_regression = LogisticRegression(max_iter=200)
logistic_regression_cv = gridSearch.runGridSearchWithCustomScorer(X, y, logistic_regression, params)

print("tuned hyperparameters :(best parameters) ",logistic_regression_cv.best_params_)
print("best score :",logistic_regression_cv.best_score_)

# Not going to run the above. It takes about 2 minutes. But this is the output:
# tuned hyperparameters :(best parameters)  {'C': np.float64(0.01), 'penalty': 'l2'}
# best score : 36444.7


