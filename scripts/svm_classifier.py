import os

import joblib
from sklearn.svm import LinearSVC


def train_svm_classifier(X_train, y_train):
    """
    Train a Support Vector Machine (SVM) classifier and save it to a file.

    This function trains an SVM classifier on the provided training data and labels.
    After training, the classifier is serialized and saved to a file in the '../classifiers/' directory.

    :param X_train: numpy.ndarray
        The training data features. Shape should be (n_frames, n_mels).
    :param y_train: numpy.ndarray
        The target labels. Shape should be (n_frames,).
    :return: None
        The trained SVM classifier is saved to a file in the '../classifiers/' directory.
    """
    # Create the SVM classifier.
    clf = LinearSVC(random_state=42)

    # Train the SVM classifier.
    clf.fit(X_train, y_train)

    # Save the SVM classifier to a file.
    path = '../classifiers/svm_classifier.joblib'
    joblib.dump(clf, path)
    print(f'SVM classifier saved to {os.getcwd().replace("\\", "/")}/{path}\n')
