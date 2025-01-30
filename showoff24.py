import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from utils import filteredModuleData
from utils.expectedValue import expectedValue

X, y = filteredModuleData.getPassedTestModelData()

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2, test_size=3000)

baseline = DummyClassifier().fit(x_train, y_train)
baseline_score = expectedValue(y_test, baseline.predict(x_test))
print(baseline_score)

scores = []
labels = []
def addScore(model, label):
    score = expectedValue(y_test, model.predict(x_test))
    scores.append(score)
    labels.append(label)


logistic_regression_model = LogisticRegression(C=np.float64(0.01), penalty='l2', max_iter=200).fit(x_train, y_train)
addScore(logistic_regression_model, 'Logistic Regression')

naive_bayes_model = GaussianNB().fit(x_train, y_train)
addScore(naive_bayes_model, 'Naive Bayes')

lda_model = LinearDiscriminantAnalysis(store_covariance=True, solver='lsqr', shrinkage=np.float64(0.1)).fit(x_train, y_train)
addScore(lda_model, 'LDA')

qda_model = QuadraticDiscriminantAnalysis(store_covariance=True, reg_param=np.float64(0.0)).fit(x_train, y_train)
addScore(qda_model, 'QDA')

bagging_model = BaggingClassifier(max_samples=0.05).fit(x_train, y_train)
addScore(bagging_model, 'Bagging')

random_forest = RandomForestClassifier(criterion='entropy', max_features='sqrt').fit(x_train, y_train)
addScore(random_forest, 'Random Forest')

linear_svc = LinearSVC(fit_intercept=True, C=0.01, loss='hinge', penalty='l2').fit(x_train, y_train)
addScore(linear_svc, 'Linear SVC')

fig, ax = plt.subplots(figsize=(12,12))

ax.bar(labels, scores)
ax.set_xlabel('Model')
ax.set_ylabel('Expected value')
plt.axhline(y=baseline_score, color='black', ls='--')

print(f'Logistic regression score {scores[0]}')
print(f'Linear SVC score {scores[6]}')

plt.show()
