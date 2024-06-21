import random as rnd

import numpy as np

from dataset import load_audio_clips
from feature_extraction import extract_features


def shuffle_audio_clips_and_labels(audio_clips, labels):
    """
    Shuffle the audio clips and labels in the same order.

    This function shuffles the audio clips and labels in the same order, so that the audio clips and labels
    remain aligned after shuffling.

    Returns the shuffled audio clips and labels as NumPy arrays.

    :param audio_clips: list
        The list of audio clips.
    :param labels: list
        The list of labels.
    :return: tuple
        A tuple containing the shuffled audio clips and labels as NumPy arrays.
        - numpy.ndarray: The shuffled audio clips.
        - numpy.ndarray: The shuffled labels.
    """
    combined = list(zip(audio_clips, labels))
    rnd.shuffle(combined)
    audio_clips, labels = zip(*combined)

    return np.array(audio_clips), np.array(labels)


if __name__ == '__main__':
    # Load the audio clips
    foreground_train_audio_clips, foreground_dev_audio_clips, foreground_test_audio_clips, \
        background_train_audio_clips, background_dev_audio_clips, background_test_audio_clips = load_audio_clips()

    # Create labels, where foreground is 1 and background is 0
    foreground_train_labels = [1] * len(foreground_train_audio_clips)
    background_train_labels = [0] * len(background_train_audio_clips)
    foreground_dev_labels = [1] * len(foreground_dev_audio_clips)
    background_dev_labels = [0] * len(background_dev_audio_clips)

    # Combine the foreground and background audio clips and labels
    train_audio_clips = foreground_train_audio_clips + background_train_audio_clips
    train_labels = foreground_train_labels + background_train_labels
    dev_audio_clips = foreground_dev_audio_clips + background_dev_audio_clips
    dev_labels = foreground_dev_labels + background_dev_labels

    # Shuffle the combined audio clips and labels for better model training
    train_audio_clips, train_labels = shuffle_audio_clips_and_labels(train_audio_clips, train_labels)
    dev_audio_clips, dev_labels = shuffle_audio_clips_and_labels(dev_audio_clips, dev_labels)

    # Extract features from the audio clips
    train_features = np.array([extract_features(audio_clip) for audio_clip in train_audio_clips])
    dev_features = np.array([extract_features(audio_clip) for audio_clip in dev_audio_clips])
