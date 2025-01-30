import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

import utils.expectedValue
from utils import filteredModuleData, modelMetrics

X, y = filteredModuleData.getPassedTestModelData()

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2)
model = LinearDiscriminantAnalysis(store_covariance=True, solver='lsqr', shrinkage=np.float64(0.1)).fit(x_train, y_train)

modelMetrics.showSplitGraph(x_test, y_test, model)