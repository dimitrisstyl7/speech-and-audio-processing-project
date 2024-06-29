import numpy as np

from dataset import load_train_audio_clips
from feature_extraction import extract_features
from mlp_classifier import train_mlp_classifier
from rnn_classifier import train_rnn_classifier
from svm_classifier import train_svm_classifier


def main():
    """
    Entry point for training audio classifiers.

    This function performs the following steps:
    1. Loads foreground and background audio clips for training.
    2. Extracts features from the audio clips using `extract_features`.
    3. Prepares labels where 1 indicates foreground and 0 indicates background.
    4. Flattens features and labels for compatibility with classifiers.
    5. Concatenates flattened features and labels to form training datasets.
    6. Trains multiple classifiers: MLP (Multi-Layer Perceptron), SVM (Support Vector Machine),
       and RNN (Recurrent Neural Network).
       - MLP and SVM classifiers use flattened features.
       - RNN classifier requires 3D features and 2D labels, prepared by concatenating and
         transposing initial features and labels.
    :return: None
    """
    # Load the train audio clips.
    print('\nLoading train audio clips...\n')
    foreground_train_audio_clips, background_train_audio_clips = load_train_audio_clips()

    # Extract features from the train audio clips.
    print('Extracting features...\n')
    foreground_train_features = np.array([extract_features(audio_clip) for audio_clip in foreground_train_audio_clips])
    background_train_features = np.array([extract_features(audio_clip) for audio_clip in background_train_audio_clips])

    # Create labels for each foreground and background train audio clip, where 1 is foreground and 0 is background.
    foreground_train_labels = np.array([np.ones(audio_clip.shape[1]) for audio_clip in foreground_train_features])
    background_train_labels = np.array([np.zeros(audio_clip.shape[1]) for audio_clip in background_train_features])

    # Flatten the train features so that each row is a frame.
    foreground_train_features_flattened = foreground_train_features.reshape(-1, foreground_train_features.shape[1])
    background_train_features_flattened = background_train_features.reshape(-1, background_train_features.shape[1])

    # Flatten the train labels to match the flattened train features.
    foreground_train_labels_flattened = foreground_train_labels.reshape(-1)
    background_train_labels_flattened = background_train_labels.reshape(-1)

    # Combine the foreground and background flattened train features and labels.
    train_features = np.concatenate((foreground_train_features_flattened, background_train_features_flattened))
    train_labels = np.concatenate((foreground_train_labels_flattened, background_train_labels_flattened))

    print('Classifiers training started... This may take a while.\n')

    # MLP (Multi-Layer Perceptron) classifier training.
    print('Training MLP classifier...')
    train_mlp_classifier(train_features, train_labels)

    # SVM (Support Vector Machine) classifier training.
    print('Training SVM classifier...')
    train_svm_classifier(train_features, train_labels)

    # RNN (Recurrent Neural Network) classifier training.
    print('Training RNN classifier...')

    # RNN classifier needs a 3D input (X_train), so we don't need the flattened features; instead, we can use
    # the initial features. Also, the RNN classifier needs the input shape to be (n_samples, n_frames, n_mels),
    # where n_samples is the number of train audio clips, n_frames is the number of frames in each audio clip, and
    # n_mels is the number of features (melspectrogram bins) in each frame. We can achieve this by concatenating
    # and transposing the foreground and background train features.
    train_features = np.concatenate((foreground_train_features, background_train_features)).transpose(0, 2, 1)

    # Due to the 3D input of train features, we don't need the flattened labels. Instead, the labels shape should be
    # (n_samples, n_frames), where n_samples is the number of train audio clips and n_frames is the number of frames.
    # We can achieve this by concatenating the foreground and background train labels.
    train_labels = np.concatenate((foreground_train_labels, background_train_labels))

    train_rnn_classifier(train_features, train_labels)


if __name__ == '__main__':
    main()
