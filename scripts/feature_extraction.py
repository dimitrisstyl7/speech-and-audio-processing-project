import librosa
import numpy as np
from librosa import feature

from common import sr, n_fft, hop_length


def extract_features(audio_data):
    """
    Compute the Mel-frequency spectrogram of the audio data and convert it to decibels units.

    This function computes the Mel-frequency spectrogram of the input audio data using specified parameters
    (sample rate, window length, hop length, number of Mel bands), and converts it to decibels units
    using logarithmic scaling.

    Returns the Mel-frequency spectrogram in decibels units, representing the extracted features.

    :param audio_data: numpy.ndarray or array-like
        The input audio data as a 1-dimensional array.
    :return: numpy.ndarray
        The computed Mel-frequency spectrogram in decibels.
    """
    melspectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=96)
    return librosa.power_to_db(melspectrogram, ref=np.max)
