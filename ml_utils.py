import typing

import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics
import numpy as np

ModelClassifier = typing.Union[
    sklearn.linear_model.LogisticRegression,
    sklearn.ensemble.RandomForestClassifier,
]


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: ModelClassifier,
) -> None:
    """Function which fits sklearn classifier object on train data and tests on
    test data. Prints balanced accuracy to terminal window.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Testing features
        y_test (np.ndarray): Testing labels
        model (ModelClassifier): Instance of sklearn classifier model
            to be trained
    """
    model.fit(X_train, y_train, X_test, y_test)
    model.load_best_model()
    train_pred = model.predict(X_train)

    train_cmat = sklearn.metrics.confusion_matrix(
        y_true=y_train,
        y_pred=train_pred,
    )
    train_ba = np.mean(np.diag(train_cmat) / np.sum(train_cmat, axis=1))
    print(f"Final Train Balanced Accuracy: {train_ba:.3f}")

    test_pred = model.predict(X_test)
    test_cmat = sklearn.metrics.confusion_matrix(
        y_true=y_test,
        y_pred=test_pred,
    )
    test_ba = np.mean(np.diag(test_cmat) / np.sum(test_cmat, axis=1))
    print(f"Final Test Balanced Accuracy: {test_ba:.3f}")


def get_label_distribution(labels: typing.List) -> None:
    """Function which gets distribution of positive/negative classes in label
    list

    Args:
        labels (list): List of labels to analyze
    """
    n = len(labels)
    pos_labels = np.sum(labels)
    pos_pct = pos_labels / n

    neg_labels = n - pos_labels
    neg_pct = neg_labels / n

    print(
        f"Positive Labels: {pos_labels} ({pos_pct:.3f}), Negative Labels: {neg_labels} ({neg_pct:.3f})"
    )
