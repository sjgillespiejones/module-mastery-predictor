from sklearn.svm import LinearSVC
from utils import filteredModuleData, gridSearch

X, y = filteredModuleData.getPassedTestModelData()

c_params = [0.0001,0.001, 0.01, 0.1, 1]

params = [
    { "C": c_params, "penalty": ['l1', 'l2'],"loss":['hinge', 'squared_hinge']},
]
support_vector_classifier = LinearSVC(fit_intercept=True)
svc_cv = gridSearch.runGridSearchWithCustomScorer(X, y, support_vector_classifier, params)

print("tuned hyperparameters :(best parameters) ",svc_cv.best_params_)
print("best score :",svc_cv.best_score_)

# tuned hyperparameters :(best parameters)  {'C': 0.01, 'loss': 'hinge', 'penalty': 'l2'}
# best score : 36666.6