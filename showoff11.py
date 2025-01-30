import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

import utils.expectedValue
from utils import filteredModuleData, gridSearch

X, y = filteredModuleData.getPassedTestModelData()

params = {"shrinkage":np.linspace(0.1,0.9,8), "solver":['lsqr', 'eigen']}
lda_classifier = LinearDiscriminantAnalysis(store_covariance=True)
lda_classifier_cv = gridSearch.runGridSearchWithCustomScorer(X, y, lda_classifier, params)

print("tuned hyperparameters :(best parameters) ",lda_classifier_cv.best_params_)
print("best score :",lda_classifier_cv.best_score_)

# tuned hyperparameters :(best parameters)  {'shrinkage': np.float64(0.1), 'solver': 'lsqr'}
# best score : 25392.6