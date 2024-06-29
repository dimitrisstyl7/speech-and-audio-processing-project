import os

import joblib
from sklearn.neural_network import MLPClassifier


def train_mlp_classifier(X_train, y_train):
    """
    Train the Multi-Layer Perceptron (MLP) classifier and save it to a file.

    This function trains the MLP classifier on the provided training data and labels.
    After training, the classifier is serialized and saved to a file in the '../classifiers/' directory.

    :param X_train: numpy.ndarray
        The training data features. Shape should be (n_frames, n_mels).
    :param y_train: numpy.ndarray
        The target labels. Shape should be (n_frames,).
    :return: None
    """
    # Create the MLP classifier.
    clf = MLPClassifier(hidden_layer_sizes=(128, 64, 32), random_state=42, early_stopping=True,
                        validation_fraction=0.15)

    # Train the MLP classifier.
    clf.fit(X_train, y_train)

    # Save the MLP classifier to a file.
    path = '../classifiers/mlp_classifier.joblib'
    joblib.dump(clf, path)
    print(f'MLP classifier saved to {os.getcwd().replace('\\', '/')}/{path}\n')
