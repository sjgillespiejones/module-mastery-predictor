from utils import filteredModuleData
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import modelMetrics

X, y = filteredModuleData.getPassedTestModelData()
print(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2)
model = LogisticRegression(max_iter=200).fit(x_train, y_train)
modelMetrics.showSplitGraph(x_test, y_test, model)