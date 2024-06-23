import numpy as np

from dataset import load_audio_clips
from feature_extraction import extract_features

if __name__ == '__main__':
    # Load the audio clips
    foreground_audio_clips, background_audio_clips = load_audio_clips()

    # Extract features from the audio clips
    foreground_features = np.array([extract_features(audio_clip) for audio_clip in foreground_audio_clips])
    background_features = np.array([extract_features(audio_clip) for audio_clip in background_audio_clips])

    # Create labels for each foreground frame and background frame, where 1 is foreground and 0 is background
    foreground_labels = np.array([[1 for _ in range(feature.shape[1])] for feature in foreground_features])
    background_labels = np.array([[0 for _ in range(feature.shape[1])] for feature in background_features])

    # Combine the foreground and background features and labels
    features = np.concatenate((foreground_features, background_features))
    labels = np.concatenate((foreground_labels, background_labels))
