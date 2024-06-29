import json
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning messages.

import joblib
import keras
import numpy as np
from scipy import signal
from sklearn.metrics import accuracy_score

from audio import play_audio
from common import frame_duration
from dataset import load_test_audio_clip, get_test_audio_duration
from feature_extraction import extract_features

keras.utils.disable_interactive_logging()  # Disable TensorFlow interactive logging.


def get_ground_truth_labels(audio_clip_name):
    """
    Generate the ground truth label sequence for the given audio clip.

    This function reads word timestamps from a JSON file corresponding to the given audio clip name,
    and generates a sequence of labels where 0 represents background and 1 represents foreground.
    The sequence is based on the timestamps of words and the frame duration.

    :param audio_clip_name: str
        The name of the audio clip for which to generate the ground truth labels.
    :return: numpy.ndarray
        An array representing the ground truth label sequence, where 0 indicates background frames
        and 1 indicates foreground frames.
    """
    # Open the JSON file containing the word timestamps for the given audio clip.
    f = open(f'../datasets/test/timestamps/{audio_clip_name.replace('.mp3', '.json')}', 'r')
    words_timestamps = json.load(f)

    label_sequence = []
    prev_end_timestamp = 0.0

    for word_timestamps in words_timestamps:
        start_timestamp = word_timestamps[1]
        end_timestamp = word_timestamps[2]

        # Calculate the number of background frames between words.
        num_background_frames = int(round((start_timestamp - prev_end_timestamp) / frame_duration))
        if num_background_frames > 0:
            label_sequence += [0] * num_background_frames

        # Calculate the number of foreground frames for the word.
        num_foreground_frames = int(round((end_timestamp - start_timestamp) / frame_duration))
        label_sequence += [1] * num_foreground_frames

        # Update the previous end timestamp.
        prev_end_timestamp = end_timestamp

    # Calculate the remaining background frames until the end of the audio clip
    audio_clip_duration = get_test_audio_duration(audio_clip_name)
    num_remaining_background_frames = int(round((audio_clip_duration - prev_end_timestamp) / frame_duration))
    if num_remaining_background_frames > 0:
        label_sequence += [0] * num_remaining_background_frames

    return np.array(label_sequence)


def predict_labels(classifier_name, audio_clip_name):
    """
    Predict the labels for a given audio clip using a specified classifier.

    This function loads the audio clip and the specified classifier, extracts features from the audio clip,
    and then uses the classifier to predict a sequence of labels indicating foreground (1) or background (0).
    The predicted labels are smoothed using a median filter.

    :param classifier_name: str
        The name of the classifier file to be used for prediction. Should be either a '.joblib' or a
        Keras model file.
    :param audio_clip_name: str
        The name of the audio clip file to be processed.
    :return: list of int
        A list representing the predicted label sequence, where 1 indicates foreground and 0 indicates background.
    """
    # Load the test audio clip.
    test_clip = load_test_audio_clip(audio_clip_name)

    # Load the classifier model.
    clf = joblib.load(f'../classifiers/{classifier_name}') if classifier_name.endswith(
        '.joblib') else keras.models.load_model(f'../classifiers/{classifier_name}')

    # Predict the labels using the chosen classifier.
    if classifier_name.startswith('rnn'):
        # For RNN classifiers, the features need to be in the shape (n_samples, n_frames, n_mels).
        test_features = np.array([extract_features(test_clip)]).transpose(0, 2, 1)

        # Predict and threshold the results to obtain binary labels.
        label_sequence = (clf.predict(test_features) > 0.5).reshape(-1)
    else:
        # For other classifiers (MLP, SVM and Least Squares), the features are in the shape (n_frames, n_mels).
        test_features = np.array(extract_features(test_clip)).transpose(1, 0)

        # Predict the labels directly.
        label_sequence = clf.predict(test_features)

    # Convert the predicted label sequence to a list of integers.
    label_sequence = label_sequence.astype(np.int32).tolist()

    # Apply median filter to the predicted label sequence to smooth the results.
    label_sequence = signal.medfilt(label_sequence, kernel_size=5)

    return label_sequence


def extract_words_timestamps(label_sequence):
    """
    Extract word timestamps from a sequence of binary labels.

    This function processes a sequence of binary labels (where 1 indicates foreground and 0 indicates background)
    and extracts the start and end timestamps for each contiguous segment of foreground labels.

    :param label_sequence: list[int]
        A sequence of binary labels representing frames. Shape should be (n_frames,).
    :return: list[tuple[float, float]]
        A list of tuples, each containing the start and end timestamps of a word.
    """

    def get_starting_timestamp(label_idx):
        """
        Calculate the starting timestamp for a given frame index.

        :param label_idx: int
            The index of the label/frame in the sequence.
        :return: float
            The starting timestamp of the frame in seconds.
        """
        starting_timestamp = label_idx * frame_duration
        return starting_timestamp

    word_timestamps = []
    start_timestamp = 0.0
    start_timestamp_flag = False
    for idx, label in enumerate(label_sequence):
        # If the label is 0 and the start timestamp flag is True, save the word's timestamps.
        if label == 0 and start_timestamp_flag:
            start_timestamp_flag = False
            word_timestamps.append((start_timestamp, get_starting_timestamp(idx)))
            continue

        # If the label is 0, continue.
        if label == 0:
            continue

        # If the label is 1 and the starting timestamp flag is False, set the starting timestamp.
        if not start_timestamp_flag:
            start_timestamp_flag = True
            start_timestamp = get_starting_timestamp(idx)

        # If the label is 1, the current frame is the last one, and the starting timestamp flag is True,
        # save the word's timestamps.
        if idx == len(label_sequence) - 1 and start_timestamp_flag:
            word_timestamps.append((start_timestamp, get_starting_timestamp(idx)))

    return word_timestamps


def menu():
    """
    Display and operate a command-line menu for detecting words in audio clips using classifiers.

    The menu allows the user to:
    1. Choose audio clips and classifiers.
    2. Detect words in the selected audio clips using the chosen classifiers.
    3. Clear the console.
    4. Exit the program.

    :return: None
    """

    def choose_indices(input_message, max_length):
        """
        Prompt the user to choose indices from a list.

        :param input_message: str
            Message to display to the user.
        :param max_length: int
            Maximum length of the list.
        :return: list[int] or None
            List of chosen indices or None if user chooses to go back.
        """
        chosen_indices = input(f'\n> {input_message} (separated by comma): ')

        try:
            chosen_indices = [int(idx) - 1 for idx in chosen_indices.split(',')]

            # Remove duplicate indices.
            chosen_indices = list(set(chosen_indices))
        except:
            raise TypeError('Invalid input! Please try again.')

        # If -1 is in chosen_indices, return None indicating to return to the main menu.
        if -1 in chosen_indices:
            print('\n\nReturning to main menu...\n')
            return None

        # Check if chosen_indices are within valid range.
        if max(chosen_indices) >= max_length or min(chosen_indices) < 0:
            raise TypeError('Invalid input! Please try again.')

        return chosen_indices

    while True:
        try:
            print('\n=========== Word Detector Menu ===========')
            print('\n\t1. Detect words in audio clips')
            print('\t2. Clear console')
            print('\t3. Exit\n')
            print('==========================================')

            choice = input('\n> Enter your choice: ')

            if choice == '1':  # Detect words in audio clips.
                audio_clips_names = os.listdir('../datasets/test/clips')
                print('\nAvailable audio clips:')
                for idx, audio_clip_name in enumerate(audio_clips_names):
                    print(f'\t{idx + 1}. {audio_clip_name}')
                print('\t0. Back')

                # Choose audio clips.
                chosen_audio_clips_indices = choose_indices('Enter the indices of the audio clips to detect words',
                                                            len(audio_clips_names))

                # Check if chosen_audio_clips_indices is None, indicating to return to the main menu.
                if chosen_audio_clips_indices is None:
                    continue

                # Choose classifiers.
                print('\nChoose the classifier/s to use:')
                classifiers_name = os.listdir('../classifiers')
                for idx, classifier_name in enumerate(classifiers_name):
                    if classifier_name.startswith('lstsq'):
                        print(f'\t{idx + 1}. Least Squares (LS) classifier')
                    elif classifier_name.startswith('svm'):
                        print(f'\t{idx + 1}. Support Vector Machine (SVM) classifier')
                    elif classifier_name.startswith('mlp'):
                        print(f'\t{idx + 1}. Multi-Layer Perceptron (MLP) classifier')
                    elif classifier_name.startswith('rnn'):
                        print(f'\t{idx + 1}. Recurrent Neural Network (RNN) classifier')
                print('\t0. Back')

                chosen_classifiers_indices = choose_indices('Enter the indices of the classifiers to use',
                                                            len(classifiers_name))

                # Check if chosen_classifiers_indices is None, indicating to return to the main menu.
                if chosen_classifiers_indices is None:
                    continue

                # Print chosen audio clips and classifiers.
                print('\nFor the following audio clips:')
                for idx in chosen_audio_clips_indices:
                    print(f'\t+ {audio_clips_names[idx]}')

                print('\nYou have chosen the following classifiers:')
                for idx in chosen_classifiers_indices:
                    if classifiers_name[idx].startswith('lstsq'):
                        print(f'\t+ Least Squares (LS) classifier')
                    elif classifiers_name[idx].startswith('svm'):
                        print(f'\t+ Support Vector Machine (SVM) classifier')
                    elif classifiers_name[idx].startswith('mlp'):
                        print(f'\t+ Multi-Layer Perceptron (MLP) classifier')
                    elif classifiers_name[idx].startswith('rnn'):
                        print(f'\t+ Recurrent Neural Network (RNN) classifier')

                # Start detecting words in the chosen audio clips using the chosen classifiers.
                for audio_clip_idx in chosen_audio_clips_indices:
                    print(f'\n\n========== Detecting words in {audio_clips_names[audio_clip_idx]} ==========')

                    for clf_idx in chosen_classifiers_indices:
                        classifier_name = classifiers_name[clf_idx]
                        if classifier_name.startswith('lstsq'):
                            print(f'\n+ Detecting words using the LS classifier...')
                        elif classifier_name.startswith('svm'):
                            print(f'\n+ Detecting words using the SVM classifier...')
                        elif classifier_name.startswith('mlp'):
                            print(f'\n+ Detecting words using the MLP classifier...')
                        elif classifier_name.startswith('rnn'):
                            print(f'\n+ Detecting words using the RNN classifier...')

                        # Get the ground truth labels for the current audio clip.
                        ground_truth_labels = get_ground_truth_labels(audio_clips_names[audio_clip_idx])

                        # Predict the labels using the chosen classifier.
                        predicted_labels = predict_labels(classifiers_name[clf_idx], audio_clips_names[audio_clip_idx])

                        # Truncate predicted labels if they exceed ground truth labels.
                        if len(ground_truth_labels) < len(predicted_labels):
                            predicted_labels = predicted_labels[:len(ground_truth_labels)]

                        # Extract words timestamps from the predicted labels.
                        extracted_words_timestamps = extract_words_timestamps(predicted_labels)

                        # Print the number of the detected words and their timestamps.
                        print(f'Classifier detected {len(extracted_words_timestamps)} word/s.')
                        print(f'Accuracy score: {accuracy_score(ground_truth_labels, predicted_labels):.2f}')
                        print('\nDetected words\' timestamps:')
                        for idx, (start, end) in enumerate(extracted_words_timestamps):
                            print(f'\t{idx + 1}) {start:.3f} - {end:.3f} seconds')

                        # Listen specific word if desired.
                        while True:
                            word_idx = input(
                                '\n> Enter the index of the word to listen to it or any other key to continue: ')

                            try:
                                word_idx = int(word_idx) - 1
                            except ValueError:  # Invalid input, continue execution.
                                print('\nContinuing execution...')
                                break

                            # Check if the word index is valid.
                            if word_idx not in range(len(extracted_words_timestamps)):
                                print('\nContinuing execution...')
                                break

                            # Listen to the specific word.
                            start_time = extracted_words_timestamps[word_idx][0]
                            end_time = extracted_words_timestamps[word_idx][1]
                            duration = end_time - start_time
                            print(f'Listening to the word {word_idx + 1} ({duration:.3f} seconds)...')
                            play_audio(audio_clips_names[audio_clip_idx], start_time, duration)

                print('\n\nReturning to main menu...\n')

            elif choice == '2':  # Clear console.
                os.system('cls' if os.name == 'nt' else 'clear')

            elif choice == '3':  # Exit.
                print('\nExiting...\n')
                sys.exit()

            else:  # Invalid input.
                print('\nInvalid input! Please try again.\n')

        except KeyboardInterrupt:
            print('\n\nExecution interrupted. Exiting...\n')
            sys.exit()

        except FileNotFoundError as e:
            print(f'\n{e}')
            print('\n\nReturning to main menu...\n')

        except TypeError as e:
            print(f'\n{e}')
            print('\n\nReturning to main menu...\n')

        except Exception:
            print('\nSomething went wrong! Please try again.')
            print('\n\nReturning to main menu...\n')


if __name__ == '__main__':
    menu()
