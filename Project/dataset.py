import math
import os
import random as rnd

import librosa
import numpy as np
import pandas as pd

rnd.seed(42)  # Set the random seed for reproducibility.


def load_audio_clips():
    """
    Load and preprocess both foreground and background audio clips for training, development (validation), and testing.

    This function coordinates the loading and preprocessing of both foreground and background audio clips
    for the different phases ('train', 'dev', 'test') of the dataset.

    Returns six lists of audio clips:
    - Foreground audio clips: train, dev, and test sets.
    - Background audio clips: train, dev, and test sets.

    :return: tuple
        A tuple containing six lists:
        - List[np.ndarray]: Foreground train audio clips.
        - List[np.ndarray]: Foreground dev audio clips.
        - List[np.ndarray]: Foreground test audio clips.
        - List[np.ndarray]: Background train audio clips.
        - List[np.ndarray]: Background dev audio clips.
        - List[np.ndarray]: Background test audio clips.
    """
    # Load the foreground audio clips (train, dev, and test).
    foreground_train_audio_clips, foreground_dev_audio_clips, foreground_test_audio_clips, median_duration = \
        load_foreground_audio_clips()

    # Load the background audio clips (train, dev, and test).
    background_train_audio_clips, background_dev_audio_clips, background_test_audio_clips = \
        load_background_audio_clips(len(foreground_train_audio_clips), median_duration)

    return foreground_train_audio_clips, foreground_dev_audio_clips, foreground_test_audio_clips, \
        background_train_audio_clips, background_dev_audio_clips, background_test_audio_clips


def load_foreground_audio_clips():
    """
    Load foreground audio clips for train, dev, and test phases.

    :return: tuple
        A tuple containing:
        - List[numpy.ndarray]: Foreground train audio clips.
        - List[numpy.ndarray]: Foreground dev (validation) audio clips.
        - List[numpy.ndarray]: Foreground test audio clips.
        - int: Median duration in seconds of the loaded foreground train audio clips.
    """
    # Load the foreground train audio clips and calculate the median duration.
    foreground_train_audio_clips, median_duration = load_foreground_audio_clips_from_tsv('train')

    # Load the foreground dev (validation) audio clips and preprocess them based on the median duration.
    foreground_dev_audio_clips = load_foreground_audio_clips_from_tsv('dev', median_duration)

    # Load the foreground test audio clips.
    foreground_test_audio_clips = load_foreground_audio_clips_from_tsv('test')

    return foreground_train_audio_clips, foreground_dev_audio_clips, foreground_test_audio_clips, median_duration


def load_foreground_audio_clips_from_tsv(phase, median_duration=None):
    """
    Load and preprocess foreground audio clips based on clip durations and the specified phase.

    Depending on the phase:
    - For 'test': Loads and returns audio clips without preprocessing.
    - For 'train': Calculates the median duration of the audio clips and then preprocesses audio clips
      to match this median duration.
    - For 'dev': Preprocesses audio clips based on the median duration and returns them.

    :param phase: str
        The phase of the dataset. Expected values are 'train', 'dev', or 'test'.
    :param median_duration: int, optional
        The median duration in seconds of the audio clips. If None, then phase is 'train' and the median
        duration will be calculated from the audio clips' durations.

    :return:
        If phase is 'test':
            List[numpy.ndarray]: Audio clips without preprocessing.
        If phase is 'dev':
            List[numpy.ndarray]: Preprocessed audio clips.
        If phase is 'train':
            tuple: A tuple containing:
                - List[numpy.ndarray]: Preprocessed audio clips.
                - int: Median duration in seconds of the loaded foreground audio clips.
    """
    # Load the TSV file for the given phase (e.g., 'train.tsv', 'dev.tsv', 'test.tsv').
    tsv_file = load_tsv_file(f'{phase}.tsv')

    # Extract the list of audio file names from the TSV file.
    audio_names = tsv_file['path'].values.tolist()

    # Replace '.mp3' extensions with '.wav' in the audio names.
    audio_names = [audio_name.replace('.mp3', '.wav') for audio_name in audio_names]

    # For the 'test' phase, load and return audio clips without preprocessing.
    if phase == 'test':
        audio_clips = [load_audio_clip(audio_name, 'foreground') for audio_name in audio_names]
        return audio_clips

    # For the 'train' phase, calculate the median duration of the audio clips.
    if median_duration is None:  # median_duration is None indicates that the phase is 'train'.
        median_duration = calculate_median_duration(audio_names)

    # Load and preprocess audio clips to match the median duration.
    audio_clips = [load_audio_clip(audio_name, 'foreground', median_duration) for audio_name in audio_names]

    # For the 'dev' phase, return the preprocessed audio clips.
    if phase == 'dev':
        return audio_clips

    # For the 'train' phase, return the preprocessed audio clips and the median duration.
    return audio_clips, median_duration


def load_background_audio_clips(number_of_foreground_train_audio_clips, median_duration):
    """
    Load and preprocess background audio clips for training, development (validation), and testing.

    This function selects background audio clips from the '../Datasets/background/clips' directory
    and preprocesses them based on the specified `median_duration` for 'train' and 'dev' phases.

    Returns lists of preprocessed audio clips for train, dev, and test phases.

    :param number_of_foreground_train_audio_clips: int
        Number of foreground audio clips used for training.
        Determines how many background audio clips will be selected for training.
    :param median_duration: int
        The median duration in seconds used for preprocessing audio clips.
        Audio clips are padded or truncated to match this duration during preprocessing.
    :return: tuple
        A tuple containing three lists:
        - List[np.ndarray]: Preprocessed background audio clips for train.
        - List[np.ndarray]: Preprocessed background audio clips for dev.
        - List[np.ndarray]: Background audio clips for test without preprocessing.

    Notes
    -----
    - Background audio clips are assumed to be located in the '../Datasets/background/clips' directory.
    - All background audio clips are assumed to have a fixed duration of 5 seconds.
    - For the 'test' phase, audio clips are loaded without any preprocessing.
    - For 'train' and 'dev' phases, audio clips are preprocessed to match the specified `median_duration`.
    """
    # Get all background audio clips from the directory.
    background_audio_clips = os.listdir('../Datasets/background/clips')

    # Choose background audio clips for train, dev, and test.
    background_train_audio_clips = rnd.sample(background_audio_clips, number_of_foreground_train_audio_clips)
    remaining_background_audio_clips = list(set(background_audio_clips) - set(background_train_audio_clips))
    background_dev_audio_clips = rnd.sample(remaining_background_audio_clips,
                                            len(remaining_background_audio_clips) // 2)
    background_test_audio_clips = list(set(remaining_background_audio_clips) - set(background_dev_audio_clips))

    # Load the background train audio clips.
    background_train_audio_clips = [load_audio_clip(audio_name, 'background', median_duration) for
                                    audio_name in background_train_audio_clips]

    # Load the background dev audio clips.
    background_dev_audio_clips = [load_audio_clip(audio_name, 'background', median_duration) for
                                  audio_name in background_dev_audio_clips]

    # Load the background test audio clips.
    background_test_audio_clips = [load_audio_clip(audio_name, 'background') for audio_name in
                                   background_test_audio_clips]

    return background_train_audio_clips, background_dev_audio_clips, background_test_audio_clips


def load_audio_clip(audio_name, directory, median_duration=None):
    """
    Load and preprocess an audio clip based on the provided durations.

    This function loads the audio data from the '../Datasets/{directory}/clips/' directory.

    :param audio_name: str
        The name of the audio file to load.
    :param directory: str
        The directory where the audio file is located. Expected values are 'foreground' or 'background'.
    :param median_duration: int, optional
        The median duration of audio clips in seconds. If provided and `audio_duration` is less than
        `median_duration`, padding is added to match `median_duration`.
    :return: numpy.ndarray
        The audio data as a numpy array.

    Notes
    -----
    - Audio files are expected to be located in the '../Datasets/{directory}/clips/' directory.
    - For 'test' phase (when `median_duration` is None), the function loads audio clips without preprocessing.
    - For 'train' and 'dev' phases, if `audio_duration` is less than `median_duration`, the function preprocesses
      the audio clip by adding padding to match `median_duration`.
    """
    path = f'../Datasets/{directory}/clips/{audio_name}'

    # If median_duration is None, load the audio clip without preprocessing (test phase).
    if median_duration is None:
        audio_clip, _ = librosa.load(path)
        return audio_clip

    # If audio_duration is less than median_duration, preprocess the audio clip with padding (train or dev phase).
    sr = librosa.get_samplerate(path)  # sample rate
    audio_duration = librosa.get_duration(path=path, sr=sr)  # duration in seconds
    if audio_duration < median_duration:
        audio_clip = librosa.load(path)
        return add_ending_padding_to_audio_clip(audio_clip, median_duration)

    # Otherwise, load the audio clip with the specified median_duration (train or dev phase).
    audio_clip, _ = librosa.load(path, duration=median_duration)
    return audio_clip


def load_tsv_file(file_name):
    """
   Load a TSV file from a specified directory within the 'Datasets' directory and return it as a DataFrame.

   :param file_name: str
       The name of the TSV file (without the directory path).
   :return: pandas.DataFrame
       The TSV file loaded as a DataFrame.
   """
    return pd.read_csv(f'../Datasets/foreground/{file_name}', sep='\t')


def calculate_median_duration(audio_names):
    """
    Calculate and return the median duration of audio clips in seconds.

    :param audio_names: list
        A list of audio file names.
    :return: int
        The median duration of audio clips in seconds.
    """
    audio_durations = []
    for audio_name in audio_names:
        path = f'../Datasets/foreground/clips/{audio_name}'
        sr = librosa.get_samplerate(path)
        audio_duration = math.ceil(librosa.get_duration(path=path, sr=sr))
        audio_durations.append(audio_duration)

    return np.median(audio_durations)


def add_ending_padding_to_audio_clip(audio, median_duration):
    """
    Add zero padding to the end of the audio clip to make it the same duration as the median duration.

    Returns the padded audio data.

    :param audio: tuple
        A tuple containing:
        - numpy.ndarray: The audio data.
        - int: The sample rate (in Hz).
    :param median_duration: int
        The median duration of audio clips in seconds.
    :return: numpy.ndarray
        The padded audio data as a numpy array.
    """
    sr = audio[1]  # sample rate
    desired_total_samples = math.ceil(sr * median_duration)  # total samples
    current_samples = len(audio[0])  # current samples
    padding = desired_total_samples - current_samples
    padded_audio = np.pad(audio[0], (0, padding), 'constant')

    return padded_audio

