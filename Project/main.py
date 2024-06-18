import math

import numpy as np

from dataset import load_tsv_file, load_audio
from feature_extraction import extract_features


def train_models():
    # Load the train.tsv file
    train_tsv = load_tsv_file('train.tsv')

    # Get a list of all the audio names
    audio_names = train_tsv['path'].values.tolist()

    # Create a dict that maps audio names to their durations
    audio_durations = {row['clip']: row['duration[ms]'] for idx, row in clip_durations_tsv.iterrows()
                       if row['clip'] in audio_names}

    # Find the median duration of the audio clips
    median_duration = math.ceil(np.median(list(audio_durations.values())))

    # Load audio clips and preprocess them if necessary
    audio_clips = [load_audio(audio_name, audio_duration, median_duration) for audio_name, audio_duration in
                   audio_durations.items()]

    # Extract features from the audio clips
    features = [extract_features(audio_clip[0]) for audio_clip in audio_clips]


if __name__ == '__main__':
    # Load the clip_durations.tsv file
    clip_durations_tsv = load_tsv_file('clip_durations.tsv')

    # Train the models (Least Squares, SVM, RNN and MLP)
    train_models()
