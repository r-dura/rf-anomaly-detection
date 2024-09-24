import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split

def create_spectrogram(data, fs=None, nperseg=1000, noverlap=500):
    if fs is None:
        # Estimate fs based on the length of the data and the expected duration
        expected_duration = 1  # in seconds
        fs = len(iq_data) / expected_duration
    
    f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return Sxx

def prepare_data(data, test_size = 0.2):
    spectrograms = []
    for segment in data:
        low_spec = create_spectrogram(segment[:,0])
        high_spec = create_spectrogram(segment[:,1])
        combined_spec = np.maximum(low_spec, high_spec)
        spectrograms.append(combined_spec)

    spectrograms = np.array(spectrograms)

    # Normalize spectrograms

    # Reshape spectrograms for autoencoder

    train_data, test_data = train_test_split(spectrograms, test_size=test_size, random_state=42)


    return train_data, test_data
    