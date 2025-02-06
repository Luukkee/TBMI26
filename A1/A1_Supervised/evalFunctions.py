import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    correct_predictions_amount = np.sum(LPred == LTrue)
    acc = correct_predictions_amount / len(LTrue)
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    number_of_classes = len(np.unique(LTrue))
    cM = np.zeros((number_of_classes, number_of_classes))

    for i in range(len(LTrue)):
        cM[LTrue[i]][LPred[i]] += 1

    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    correct_predictions_amount = np.trace(cM)
    total_predictions = np.sum(cM)
    acc = correct_predictions_amount / total_predictions
    
    return acc
