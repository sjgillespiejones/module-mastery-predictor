import numpy as np

import utils.expectedValue
from utils import filteredModuleData, gridSearch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from utils import expectedValue

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

X, y = filteredModuleData.getPassedTestModelData()

params = { 'reg_param': np.linspace(0, 0.9, 9)}

qda_classifier = QuadraticDiscriminantAnalysis(store_covariance=True)
qda_classifier_cv = gridSearch.runGridSearchWithCustomScorer(X, y, qda_classifier, params)

print("tuned hyperparameters :(best parameters) ",qda_classifier_cv.best_params_)
print("best score :",qda_classifier_cv.best_score_)

# tuned hyperparameters :(best parameters)  {'reg_param': np.float64(0.0)}
# best score : 30956.7