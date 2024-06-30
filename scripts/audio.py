import time

import vlc


def play_audio(audio_name, start_time, duration):
    """
   Play a segment of the audio file with the given name.

   This function uses VLC to play a segment of the specified audio file starting at the given time
   and for the specified duration. If the duration is less than 0.047 seconds, the function will
   print an error message and not play the audio.

   :param audio_name: str
       The name of the audio file to be played.
   :param start_time: float
       The start time in seconds from which the audio should start playing.
   :param duration: float
       The duration in seconds for which the audio should be played. Must be at least 0.047 seconds.
   :return: None
   """
    # Create VLC MediaPlayer instance.
    media_player = vlc.MediaPlayer()

    # Check if the duration is too short to be played.
    if duration < 0.047:
        print(f'Audio duration is too short, the minimum duration is 0.047 seconds.\n'
              f'Given duration: {duration:.3f} seconds.')
        return

    # Create a new media instance with the given audio name.
    media = vlc.Media(f'../datasets/test/clips/{audio_name}')

    # Set the start time and run time of the media.
    media.add_option(f'start-time={start_time}')
    media.add_option(f'run-time={duration}')

    # Set the media to the media player.
    media_player.set_media(media)

    # Start playing the media.
    media_player.play()

    # Wait for the audio to finish playing.
    while media_player.get_state() == vlc.State.Playing:
        time.sleep(0.1)
