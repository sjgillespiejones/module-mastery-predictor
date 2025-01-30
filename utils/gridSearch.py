import numpy as np

import utils.expectedValue
from utils import filteredModuleData
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from utils import expectedValue

def runGridSearchWithCustomScorer(X, y, model, params):
    scorer = make_scorer(utils.expectedValue.expectedValue)
    grid_search_cv = GridSearchCV(model, params, scoring=scorer, cv=10, n_jobs=8)
    grid_search_cv.fit(X, y)
    return grid_search_cv