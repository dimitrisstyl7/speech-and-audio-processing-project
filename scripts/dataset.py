import os
import sys

import librosa

from common import sr


def load_train_audio_clips():
    """
    Load foreground and background train audio clips from the '../datasets/train/' directory.

    :return: tuple
        A tuple containing two lists:
        - List[np.ndarray]: Foreground train audio clips.
        - List[np.ndarray]: Background train audio clips.
    """
    # Load and return the foreground and background train audio clips.
    return load_foreground_train_audio_clips(), load_background_train_audio_clips()


def load_foreground_train_audio_clips():
    """
    Load all foreground train audio clips from the '../datasets/train/foreground/clips' directory.

    :return: list
        - List[numpy.ndarray]: Foreground train audio clips.
    """
    # Get all foreground train audio clips names from the dataset.
    foreground_train_audio_clips = os.listdir('../datasets/train/foreground/clips')

    # Load and return all foreground train audio clips from the dataset.
    return [load_train_audio_clip(audio_name, 'foreground') for audio_name in foreground_train_audio_clips]


def load_background_train_audio_clips():
    """
    Load all background train audio clips from the '../datasets/train/background/clips' directory.

    :return: list
        - List[np.ndarray]: Background train audio clips.
    """
    # Get all background train audio clips from the dataset.
    background_train_audio_clips = os.listdir('../datasets/train/background/clips')

    # Load and return all background train audio clips from the dataset.
    return [load_train_audio_clip(audio_name, 'background') for audio_name in background_train_audio_clips]


def load_train_audio_clip(audio_name, directory):
    """
    Load a train audio clip from 1.5 seconds to 3.5 seconds from the specified directory.

    We choose to load the audio clip from 1.5 seconds to 3.5 seconds to minimize the amount
    of silence at the beginning and end of the audio clip.

    This function loads the audio data from the '../datasets/train/{directory}/clips/' directory.

    :param audio_name: str
        The name of the train audio file to load.
    :param directory: str
        The directory where the train audio file is located. Expected values are 'foreground' or 'background'.
    :return: numpy.ndarray
        The train audio data as a numpy array.
    """
    audio_clip, _ = librosa.load(f'../datasets/train/{directory}/clips/{audio_name}', sr=sr, offset=1.5, duration=2)
    return audio_clip


def load_test_audio_clip(audio_name):
    """
    Load a test audio clip from the '../datasets/test/clips/' directory.

    :param audio_name: str
        The name of the test audio file to load.
    :return: numpy.ndarray
        The test audio data as a numpy array.
    """
    try:
        audio_clip, _ = librosa.load(f'../datasets/test/clips/{audio_name}', sr=sr)
        return audio_clip
    except FileNotFoundError:
        print(f'\nError: The file "{audio_name}" was not found in the "{os.getcwd().replace("\\", "/")}'
              f'/../datasets/test/clips/" directory.\n')
        sys.exit(1)


def get_test_audio_duration(audio_name):
    """
    Get the duration of a test audio clip.

    :param audio_name: str
        The name of the test audio file.
    :return: float
        The duration of the test audio clip in seconds.
    """
    audio_clip, sr = librosa.load(f'../datasets/test/clips/{audio_name}', sr=None)
    return librosa.get_duration(y=audio_clip, sr=sr)
