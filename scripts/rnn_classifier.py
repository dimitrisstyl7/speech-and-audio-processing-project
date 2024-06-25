import os

from keras import layers, Sequential, Input


def train_rnn_classifier(X_train, y_train):
    """
    Train a Recurrent Neural Network (RNN) classifier and save it to a file.

    This function trains an RNN classifier on the provided training data and labels.
    After training, the classifier is serialized and saved to a file in the '../classifiers/' directory.

    :param X_train: numpy.ndarray
        The training data features. Shape should be (n_samples, n_frames, n_features).
    :param y_train: numpy.ndarray
        The target labels. Shape should be (n_samples, n_frames).
    :return: None
    """
    input_shape = (X_train.shape[1], X_train.shape[2])  # (n_frames, n_features)

    # Create the RNN model.
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(layers.SimpleRNN(64, return_sequences=True, seed=42))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy')

    # Train the RNN model.
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.15)

    # Save the RNN model to a file.
    path = '../classifiers/rnn_classifier.keras'
    model.save(path)
    print(f'RNN classifier saved to {os.getcwd().replace("\\", "/")}/{path}')
