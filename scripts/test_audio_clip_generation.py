# Part of this code is taken from the ElevenLabs API documentation and modified to suit the needs of this project.
# url: https://elevenlabs.io/docs/api-reference/text-to-speech-with-timestamps

import base64
import json

import requests


def get_text(file_name):
    """
    Retrieve text content from a specified file.

    This function reads the contents of a text file located in the '../datasets/test/text/' directory
    based on the provided file name.

    :param file_name: str
        The name of the text file (without extension) to read.
    :return: str
        The content of the text file as a string.
    """
    fs = open(f'../datasets/test/text/{file_name}.txt', 'r')
    text = fs.read()
    fs.close()
    return text


def main():
    """
    Perform text-to-speech conversion with timestamp alignment and save results.

    This function performs the following steps:
    1. Constructs the API URL for text-to-speech conversion with timestamps.
    2. Sends a POST request to the API endpoint using provided API key and text data.
    3. Parses the JSON response to obtain audio data and alignment information.
    4. Decodes the audio from base64 format and saves it as an MP3 file.
    5. Extracts words and their corresponding timestamps from the alignment data.
    6. Saves the word timestamps to a JSON file for further analysis.

    Note:
    - Replace 'VOICE_ID' and 'ENTER_YOUR_API_KEY_HERE' with actual values.
    - 'file_name' should be the name of the file used for text-to-speech conversion.

    :return: None
    """
    VOICE_ID = 'VOICE_ID'

    YOUR_XI_API_KEY = 'ENTER_YOUR_API_KEY_HERE'

    url = f'https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/with-timestamps'

    file_name = 'FILE_NAME'

    headers = {
        'Content-Type': 'application/json',
        'xi-api-key': YOUR_XI_API_KEY
    }

    data = {
        'text': (get_text(file_name)),
        'model_id': 'eleven_multilingual_v2',
        'voice_settings': {
            'stability': 0.5,
            'similarity_boost': 0.75
        }
    }

    response = requests.post(
        url,
        json=data,
        headers=headers,
    )

    if response.status_code != 200:
        print(f'Error encountered, status: {response.status_code}, '
              f'content: {response.text}')
        quit()

    # Convert the response which contains bytes into a JSON string from utf-8 encoding.
    json_string = response.content.decode('utf-8')

    # Parse the JSON string and load the data as a dictionary.
    response_dict = json.loads(json_string)

    # The 'audio_base64' entry in the dictionary contains the audio as a base64 encoded string,
    # we need to decode it into bytes in order to save the audio as a file.
    audio_bytes = base64.b64decode(response_dict['audio_base64'])

    with open(f'../datasets/test/clips/{file_name}.mp3', 'wb') as f:
        f.write(audio_bytes)

    # The 'alignment' entry contains the mapping between input characters and their timestamps.
    alignment = response_dict['alignment']

    # Create a list of words and a list of their corresponding timestamps from the alignment dictionary.
    characters = alignment['characters']
    character_start_times_seconds = alignment['character_start_times_seconds']
    character_end_times_seconds = alignment['character_end_times_seconds']

    words_and_timestamps = []
    word = ''
    start_timestamp = 0.0
    end_timestamp = 0.0

    for idx, (char, start, end) in enumerate(
            zip(characters, character_start_times_seconds, character_end_times_seconds)):
        # If the character is a space and the word is not empty, save the word and its timestamps and continue.
        if (char == ' ') and word != '':
            words_and_timestamps.append((word, start_timestamp, end_timestamp))
            word = ''
            continue

        # If the character is a space, continue.
        if char == ' ':
            continue

        # If the character is not a space and the word is empty, set the starting timestamp.
        if word == '':
            start_timestamp = start

        # Append the character to the word and update the ending timestamp.
        word += char
        end_timestamp = end

        # If the character is the last one in the alignment, save the word and its timestamps.
        if idx == len(characters) - 1:
            words_and_timestamps.append((word, start_timestamp, end_timestamp))

    # Save the alignment to a file.
    with open(f'../datasets/test/timestamps/{file_name}.json', 'w') as f:
        json.dump(words_and_timestamps, f, indent=4)


if __name__ == '__main__':
    main()
