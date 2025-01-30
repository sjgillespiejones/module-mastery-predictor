import numpy as np

import utils.expectedValue
from utils import filteredModuleData
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from utils import modelMetrics

X, y = filteredModuleData.getPassedTestModelData()

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2)
best_model = LogisticRegression(C=np.float64(0.01), penalty='l2', max_iter=200).fit(x_train, y_train)
modelMetrics.showSplitGraph(x_test, y_test, best_model)