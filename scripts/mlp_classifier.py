import os

import joblib
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(128, 64, 32), random_state=42, early_stopping=True)


def train_mlp_classifier(X_train, y_train):
    clf.fit(X_train, y_train)
    path = '../classifiers/mlp_classifier.joblib'
    joblib.dump(clf, path)
    print(f'MLP classifier saved to {os.getcwd().replace('\\', '/')}/{path}')
