import numpy as np
import os
from tqdm import tqdm 

def read_complex_csv(file_path):
    data = np.fromfile(file_path, sep=',', dtype=np.float32)
    
    # Reshape into complex numbers (I and Q interleaved)
    complex_data = data[::2] + 1j * data[1::2]
    return complex_data

def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

def combine_high_low(high_file, low_file, window_size=10000):
    high_data = read_complex_csv(high_file)
    low_data = read_complex_csv(low_file)

    high_data = normalize_data(high_data)
    low_data = normalize_data(low_data)

    min_length = min(len(high_data), len(low_data))
    num_windows = min_length // window_size
    
    combined_data = []
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = np.stack((low_data[start:end], high_data[start:end]), axis=-1)
        combined_data.append(window)

    combined_data = np.array(combined_data)
    return combined_data

def load_all_segments(base_path, num_segments=5):
    all_data = []
    for i in tqdm(range(num_segments)):
        high_file = os.path.join(base_path, f'00000H_{i}.csv')
        low_file = os.path.join(base_path, f'00000L_{i}.csv')
        segment_data = combine_high_low(high_file, low_file)
        all_data.append(segment_data)
    return np.concatenate(all_data, axis=0)

if __name__ == "__main__":
    base_path = '../data/raw/known_signal_subset'
    data = load_all_segments(base_path)
    np.save('../data/processed/all_segments.npy', data)
    print("Data saved to ../data/processed/all_segments.npy")
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")