import os

import joblib
from sklearn.svm import LinearSVC

clf = LinearSVC(random_state=42)


def train_svm_classifier(X_train, y_train):
    """
    Train the SVM classifier and save it to a file.

    :param X_train: numpy.ndarray
        The training data features. Shape should be (n_samples, n_features).
    :param y_train: numpy.ndarray
        The target labels. Shape should be (n_samples,).
    :return: None
    """
    clf.fit(X_train, y_train)
    path = '../classifiers/svm_classifier.joblib'
    joblib.dump(clf, path)
    print(f'SVM classifier saved to {os.getcwd().replace("\\", "/")}/{path}')
