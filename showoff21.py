from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from utils import filteredModuleData, gridSearch

X, y = filteredModuleData.getPassedTestModelData()
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2, train_size=10_000)

c_params = [0.01]
# gamma_params = [0.001, 0.1, 1, 10, 100, 'scale', 'auto']
gamma_params = [0.1, 1, 10, 'scale', 'auto']
params = [
    { "C": c_params, "kernel": ["poly"], "degree": [2, 3, 4, 5], "gamma": gamma_params},
    { "C": c_params, "kernel": ["rbf", "sigmoid"], "gamma": gamma_params},
]

support_vector_classifier = SVC(tol=0.01, cache_size=2000)
svc_cv = gridSearch.runGridSearchWithCustomScorer(x_train, y_train, support_vector_classifier, params)

print("tuned hyperparameters :(best parameters) ",svc_cv.best_params_)
print("best score :",svc_cv.best_score_)
