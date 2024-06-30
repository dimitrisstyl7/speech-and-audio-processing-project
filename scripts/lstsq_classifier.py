import os

import joblib
import numpy as np


class LeastSquaresClassifier:
    def __init__(self, save_path='../classifiers/lstsq_classifier.joblib'):
        """
        Initialize the Least Squares (LS) classifier.

        :param save_path: str, optional
            File path where the trained classifier weights will be saved. Defaults to
            '../classifiers/lstsq_classifier.joblib'.
        """
        self.W = None
        self.save_path = save_path

    def train(self, X_train, y_train):
        """
        Train the Least Squares (LS) classifier using the provided training data.

        This method computes the optimal weights using the normal equation and saves
        the trained classifier to the specified file path.

        :param X_train: numpy.ndarray
            Training data features of shape (n_frames, n_mels).
        :param y_train: numpy.ndarray
            Training data labels of shape (n_frames,).
        :return: None
        """
        # Add a bias term (intercept) to the features.
        X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

        # Compute weights using the normal equation.
        self.W = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train

        # Save the Least Squares classifier to the specified file path.
        joblib.dump(self, self.save_path)
        print(f'Least Squares classifier saved to {os.getcwd().replace("\\", "/")}/{self.save_path}\n')

    def predict(self, X_test):
        """
        Predict labels using the weights learned during training.

        :param X_test: numpy.ndarray
            Test data features of shape (n_frames, n_mels).
        :return: numpy.ndarray
            Predicted labels for the test data of shape (n_frames,).
        """
        # Add a bias term (intercept) to the test data.
        X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

        # Make predictions using the learned weights.
        y_pred = X_test_bias @ self.W

        # Apply a threshold to the predictions to obtain the final labels.
        y_pred[y_pred >= 0.5] = 1

        return y_pred


def train_lstsq_classifier(X_train, y_train):
    """
    Train the Least Squares classifier (LS) using the provided training data.

    This function initializes a LeastSquaresClassifier instance and trains it using the provided training data.

    :param X_train: numpy.ndarray
        Training data features of shape (n_frames, n_mels).
    :param y_train: numpy.ndarray
        Training data labels of shape (n_frames,).
    :return: None
    """
    LeastSquaresClassifier().train(X_train, y_train)
