import os

import joblib
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(128, 64, 32), random_state=42, early_stopping=True)


def train_mlp_classifier(X_train, y_train):
    """
    Train the MLP classifier and save it to a file.

    :param X_train: numpy.ndarray
        The training data features. Shape should be (n_samples, n_features).
    :param y_train: numpy.ndarray
        The target labels. Shape should be (n_samples,).
    :return: None
    """
    clf.fit(X_train, y_train)
    path = '../classifiers/mlp_classifier.joblib'
    joblib.dump(clf, path)
    print(f'MLP classifier saved to {os.getcwd().replace('\\', '/')}/{path}')
