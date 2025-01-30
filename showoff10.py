import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from utils import filteredModuleData, modelMetrics

X, y = filteredModuleData.getPassedTestModelData()

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2)
model = GaussianNB().fit(x_train, y_train)
modelMetrics.showSplitGraph(x_test, y_test, model)