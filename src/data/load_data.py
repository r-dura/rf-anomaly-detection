import numpy as np
import os
from scipy import signal
from tqdm import tqdm 

def read_complex_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = np.fromfile(file_path, sep=',', dtype=np.float32)
    # the syntax "data[::2]" means "every 2nd value in 
    # array starting from the start until the end"
    complex_data = data[::2] + 1j * data[1::2]
    return complex_data

def create_spectrogram(data, fs=1e6, nperseg=512, noverlap=256):
    f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Sxx)

def prepare_data(base_path, num_segments=40, window_size=1000, stride=500):
    all_data = []
    
    for i in tqdm(range(num_segments)):
        high_file = os.path.join(base_path, f'00000H_{i}.csv')
        low_file = os.path.join(base_path, f'00000L_{i}.csv')
        
        high_data = read_complex_csv(high_file)
        low_data = read_complex_csv(low_file)
        
        combined_data = high_data + low_data
        
        # Use sliding window to create more samples
        for start in range(0, len(combined_data) - window_size, stride):
            window = combined_data[start:start+window_size]
            spectrogram = create_spectrogram(window)
            all_data.append(spectrogram)
    
    all_data = np.array(all_data)
    
    # Normalize the data
    all_data = 2 * (all_data - np.min(all_data)) / (np.max(all_data) - np.min(all_data)) - 1
    
    # Reshape for the autoencoder (add channel dimension)
    all_data = all_data.reshape((*all_data.shape, 1))

    return all_data

if __name__ == "__main__":
    base_path = '../data/raw/known_signal_subset'
    data = load_all_segments(base_path)
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")