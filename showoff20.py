from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from utils import filteredModuleData, modelMetrics

X, y = filteredModuleData.getPassedTestModelData()
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2)
model = LinearSVC(fit_intercept=True, C=0.01, loss='hinge', penalty='l2').fit(x_train, y_train)
modelMetrics.showSplitGraph(x_test, y_test, model)
