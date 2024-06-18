import math

import librosa
import numpy as np
import pandas as pd


def load_tsv_file(file_name):
    """
    Load a TSV file located in the 'Dataset' directory and return it as a DataFrame.

    :param file_name: str
        The name of the TSV file (without the directory path).
    :return: pandas.DataFrame
        The TSV file loaded as a DataFrame.
    """
    return pd.read_csv(f'..\\Dataset\\{file_name}', sep='\t')


def load_audio(audio_name, audio_duration, median_duration):
    """
    Load the audio file with the given name and median duration.

    If the duration of the audio clip (in milliseconds) is less than the median duration,
    add padding to the end of the audio clip to match the median duration.

    Returns a tuple of the audio data and the sample rate.

    :param audio_name: str
        The name of the audio file.
    :param audio_duration: int
        The duration of the audio clip in milliseconds.
    :param median_duration: int
        The median duration of audio clips in milliseconds.
    :return: tuple
        A tuple containing:
        - numpy.ndarray: The audio data.
        - int: The sample rate (in Hz).
    """
    if audio_duration < median_duration:
        audio = librosa.load(f'..\\Dataset\\clips\\{audio_name}')
        return add_ending_padding_to_audio(audio, audio_duration, median_duration)
    return librosa.load(f'..\\Dataset\\clips\\{audio_name}', duration=median_duration / 1000)


def add_ending_padding_to_audio(audio, audio_duration, median_duration):
    """
    Add zero padding to the end of the audio clip to make it the same duration as the median duration.

    Returns a tuple of the padded audio data and the sample rate.

    :param audio: tuple
        A tuple containing:
        - numpy.ndarray: The audio data.
        - int: The sample rate (in Hz).
    :param audio_duration: int
        The duration of the audio clip in milliseconds.
    :param median_duration: int
        The median duration of audio clips in milliseconds.
    :return: tuple
        A tuple containing:
        - numpy.ndarray: The padded audio data.
        - int: The sample rate (in Hz).
    """
    sr = audio[1]  # sample rate
    desired_total_samples = math.ceil(sr * median_duration / 1000)  # total samples
    current_samples = math.ceil(sr * audio_duration / 1000)  # current audio samples
    padding = desired_total_samples - current_samples
    padded_audio = np.pad(audio[0], (0, padding), 'constant')
    return padded_audio, sr

