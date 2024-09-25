import numpy as np
import os
from scipy import signal
from tqdm import tqdm 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def read_complex_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = np.fromfile(file_path, sep=',', dtype=np.float32)
    complex_data = data[::2] + 1j * data[1::2]
    return complex_data

def create_spectrogram(data, fs=1e6, nperseg=512, noverlap=256):
    f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, np.abs(Sxx)

def process_signal(Sxx):
    log_spec = np.log1p(Sxx)
    return (log_spec - np.min(log_spec)) / (np.max(log_spec) - np.min(log_spec))

def prepare_data(base_path, num_segments=40, window_size=1000, stride=500):
    all_data = []
    
    for i in tqdm(range(num_segments)):
        high_file = os.path.join(base_path, f'00000H_{i}.csv')
        low_file = os.path.join(base_path, f'00000L_{i}.csv')
        
        high_data = read_complex_csv(high_file)
        low_data = read_complex_csv(low_file)
        
        # Combine high and low data
        combined_data = high_data + low_data
        
        for start in range(0, len(combined_data) - window_size, stride):
            window = combined_data[start:start+window_size]
            f, t, Sxx = create_spectrogram(window)
            processed_spec = process_signal(Sxx)
            all_data.append((f, t, processed_spec))
    
    return all_data

def plot_spectrogram(spec, title):
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(spec[1], spec[0], spec[2], shading='gouraud', norm=LogNorm(vmin=1e-6, vmax=1))
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Normalized Power')
    plt.ylim(0, 5e5)  # Adjust this based on your frequency range of interest
    plt.show()

def plot_distribution(data, title):
    flat_data = np.concatenate([spec[2].flatten() for spec in data])
    plt.figure(figsize=(10, 6))
    plt.hist(flat_data, bins=100, edgecolor='black')
    plt.title(title)
    plt.xlabel('Normalized Power')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.show()