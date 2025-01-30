from sklearn import metrics
import matplotlib.pyplot as plt
from utils import expectedValue
import pandas as pd
import numpy as np

def showSplitGraph(x_test, y_test, fittedModel):

    predictions = fittedModel.predict(x_test)
    confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    print(confusion_matrix)
    confusion = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
    confusion.plot()
    plt.show()
    print(metrics.classification_report(y_test, predictions))
    print(f'Score: {expectedValue.expectedValue(y_test, predictions)}')

def printTreeFeatureImportance(X, fittedModel):
    feature_names = list(X.columns)
    feature_importances = np.mean([
        tree.feature_importances_ for tree in fittedModel.estimators_
    ], axis=0)
    feature_imp = pd.DataFrame({'importance': feature_importances}, index=feature_names)
    print(feature_imp.sort_values(by='importance', ascending=False))