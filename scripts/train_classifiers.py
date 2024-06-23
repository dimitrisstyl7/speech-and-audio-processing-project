import numpy as np

from dataset import load_audio_clips
from feature_extraction import extract_features
from mlp_classifier import train_mlp_classifier


def main():
    # Load the audio clips
    print('Loading audio clips...')
    foreground_audio_clips, background_audio_clips = load_audio_clips()

    # Extract features from the audio clips
    print('Extracting features...')
    foreground_features = np.array([extract_features(audio_clip) for audio_clip in foreground_audio_clips])
    background_features = np.array([extract_features(audio_clip) for audio_clip in background_audio_clips])

    # Flatten the features so that each row is a frame
    foreground_features_flattened = foreground_features.reshape(-1, 96)
    background_features_flattened = background_features.reshape(-1, 96)

    # Create labels for each foreground frame and background frame, where 1 is foreground and 0 is background
    foreground_labels = np.ones(foreground_features_flattened.shape[0])
    background_labels = np.zeros(background_features_flattened.shape[0])

    # Combine the foreground and background flattened features and labels
    features = np.concatenate((foreground_features_flattened, background_features_flattened))
    labels = np.concatenate((foreground_labels, background_labels))

    print('Classifiers training started... This may take a while.')

    # MLP (Multi-Layer Perceptron) classifier training
    print('Training MLP classifier...')
    train_mlp_classifier(features, labels)


if __name__ == '__main__':
    main()
