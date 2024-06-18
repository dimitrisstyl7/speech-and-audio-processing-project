import librosa
import numpy as np
from librosa import feature


def extract_features(audio_data):
    """
    Compute the Mel-frequency spectrogram of the audio data and convert it to decibels units.

    This function computes the Mel-frequency spectrogram of the input audio data using specified
    parameters (window length, hop length, number of Mel bands), and converts it to decibels units
    using logarithmic scaling.

    Returns the Mel-frequency spectrogram in decibels units, representing the extracted features.

    :param audio_data: numpy.ndarray or array-like
        The input audio data as a 1-dimensional array.
    :return: numpy.ndarray
        The computed Mel-frequency spectrogram in decibels.
"""
    n_fft = 1024  # Window length
    hop_length = n_fft // 2  # Number of samples between successive frames
    n_mels = 96  # Number of Mel bands to generate (filter bank size)
    melspectrogram = librosa.feature.melspectrogram(y=audio_data, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(melspectrogram)

