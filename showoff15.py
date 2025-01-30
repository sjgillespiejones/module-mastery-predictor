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

from sklearn.ensemble import BaggingClassifier

X, y = filteredModuleData.getPassedTestModelData()

params = {
    'max_samples' : [0.05, 0.1, 0.2, 0.5]
}

bagging_classifier = BaggingClassifier()
bagging_classifier_cv = gridSearch.runGridSearchWithCustomScorer(X, y, bagging_classifier, params)

print("tuned hyperparameters :(best parameters) ",bagging_classifier_cv.best_params_)
print("best score :",bagging_classifier_cv.best_score_)

# tuned hyperparameters :(best parameters)  {'max_samples': 0.05}
# best score : 51531.2