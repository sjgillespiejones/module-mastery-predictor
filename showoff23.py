from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from utils import filteredModuleData, modelMetrics

X, y = filteredModuleData.getPassedTestModelData()

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2)
model = DummyClassifier().fit(x_train, y_train)
modelMetrics.showSplitGraph(x_test, y_test, model)