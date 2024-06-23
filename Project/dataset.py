import os

import librosa


def load_audio_clips():
    """
    Load foreground and background audio clips from the '../datasets/' directory.

    :return: tuple
        A tuple containing two lists:
        - List[np.ndarray]: Foreground audio clips.
        - List[np.ndarray]: Background audio clips.
    """
    # Load and return the foreground and background audio clips.
    return load_foreground_audio_clips(), load_background_audio_clips()


def load_foreground_audio_clips():
    """
    Load all foreground audio clips from the '../datasets/foreground/clips' directory.

    :return: list
        - List[numpy.ndarray]: Foreground audio clips.
    """
    # Get all foreground audio clips names from the dataset.
    foreground_audio_clips = os.listdir('../datasets/foreground/clips')

    # Load and return all foreground audio clips from the dataset.
    return [load_audio_clip(audio_name, 'foreground') for audio_name in foreground_audio_clips]


def load_background_audio_clips():
    """
    Load all background audio clips from the '../datasets/background/clips' directory.

    :return: list
        - List[np.ndarray]: Background audio clips.
    """
    # Get all background audio clips from the dataset.
    background_audio_clips = os.listdir('../datasets/background/clips')

    # Load and return all background audio clips from the dataset.
    return [load_audio_clip(audio_name, 'background') for audio_name in background_audio_clips]


def load_audio_clip(audio_name, directory):
    """
    Load an audio clip from 1.5 seconds to 3.5 seconds from the specified directory.

    This function loads the audio data from the '../datasets/{directory}/clips/' directory.

    :param audio_name: str
        The name of the audio file to load.
    :param directory: str
        The directory where the audio file is located. Expected values are 'foreground' or 'background'.
    :return: numpy.ndarray
        The audio data as a numpy array.

    Notes
    -----
    - Audio files are expected to be located in the '../datasets/{directory}/clips/' directory.
    """
    audio_clip, _ = librosa.load(f'../datasets/{directory}/clips/{audio_name}', offset=1.5, duration=2)
    return audio_clip
