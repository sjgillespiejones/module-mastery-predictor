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

from sklearn.ensemble import RandomForestClassifier

X, y = filteredModuleData.getPassedTestModelData()

params = {
    "criterion" : ["gini", "entropy", "log_loss"],
    "max_features": ["sqrt", "log2"]
}

random_forest_classifier = RandomForestClassifier(bootstrap=False)
random_forest_classifier_cv = gridSearch.runGridSearchWithCustomScorer(X, y, random_forest_classifier, params)

print("tuned hyperparameters :(best parameters) ", random_forest_classifier_cv.best_params_)
print("best score :", random_forest_classifier_cv.best_score_)

# tuned hyperparameters :(best parameters)  {'criterion': 'entropy', 'max_features': 'sqrt'}
# best score : 56575.2