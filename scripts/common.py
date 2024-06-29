"""
Common constants used in the project.
"""
sr = 22050  # Sample rate (Hz)
n_fft = 1024  # Window length
hop_length = n_fft // 2  # Number of samples between successive frames
frame_duration = hop_length / sr  # Duration of each frame
