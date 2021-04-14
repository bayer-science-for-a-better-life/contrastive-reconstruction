from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from utils.metrics import score_cat_acc_kaggle, score_kappa_kaggle
from collections import defaultdict
import tensorflow as tf
from sklearn.multiclass import OneVsRestClassifier
from tensorflow.keras.metrics import sparse_top_k_categorical_accuracy


def compute_embeddings(embedding_model, dataset):
    embeddings = []
    labels = []
    for x, y in tqdm(dataset):
        embeddings.append(tf.reshape(embedding_model.predict(x), (x.shape[0], -1)))
        labels.append(y.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def linear_evaluation(embedding_model, train, test, type='categorical'):
    x_train, y_train = compute_embeddings(embedding_model, train)
    x_test, y_test = compute_embeddings(embedding_model, test)
    return linear_evaluation_with_embeddings(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, type=type)


def linear_evaluation_with_embeddings(x_train, y_train, x_test, y_test, type='categorical', logistic_params=None):
    if logistic_params is None:
        logistic_params = {}
    if type == 'categorical':
        clf = LogisticRegression(random_state=0, max_iter=10000, **logistic_params).fit(x_train, y_train)
        predictions = clf.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        result = {'accuracy': accuracy}
        y_pred = clf.predict_proba(x_test)
        for k in [1, 3, 5, 10]:
            result[f'top_{k}_accuracy'] = sparse_top_k_categorical_accuracy(y_test, y_pred, k=k).numpy().mean().item()
    elif type == 'diabetic':
        values = defaultdict(list)
        for _ in range(1):
            clf = OneVsRestClassifier(
                LogisticRegression(random_state=0, max_iter=10000, class_weight=None, **logistic_params)).fit(x_train,
                                                                                                              y_train)
            predictions = clf.predict_proba(x_test)
            values['kappa_kaggle'].append(score_kappa_kaggle(y_test, predictions))
            values['cat_acc_kaggle'].append(score_cat_acc_kaggle(y_test, predictions))
        result = {key: np.mean(vals) for key, vals in values.items()}
    else:
        raise ValueError('Unknown linear type')

    return result
