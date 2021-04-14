from sklearn import metrics
import numpy as np


def transform_multilabel_to_continuous(y, threshold):
    assert isinstance(y, np.ndarray), "invalid y"
    y = y > threshold
    y = y.astype(int).sum(axis=1) - 1
    return y


def score_cat_acc(y, y_pred):
    return metrics.accuracy_score(y, y_pred).astype(np.float32)


def score_kappa(y, y_pred, labels=None):
    return metrics.cohen_kappa_score(y, y_pred, labels=labels, weights="quadratic").astype(np.float32)


def score_kappa_kaggle(y, y_pred, threshold=0.5):
    y = transform_multilabel_to_continuous(y, threshold)
    y_pred = transform_multilabel_to_continuous(y_pred, threshold)
    return score_kappa(y, y_pred)


def score_cat_acc_kaggle(y, y_pred, threshold=0.5):
    y = transform_multilabel_to_continuous(y, threshold)
    y_pred = transform_multilabel_to_continuous(y_pred, threshold)
    return score_cat_acc(y, y_pred)
