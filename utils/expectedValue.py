from sklearn.metrics import confusion_matrix

def expectedValue(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0,1])
    true_negatives = matrix[0][0]
    false_positives = matrix[0][1]
    false_negatives = matrix[1][0]
    true_positives = matrix[1][1]
    return 10 * true_negatives + true_positives - 10 * false_negatives