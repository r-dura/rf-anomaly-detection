import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(combined_data, test_size=0.2, random_state=42):
    # Separate background (normal) data and drone (anomaly) data
    background_data = combined_data[combined_data.iloc[:, -1] == 0]
    drone_data = combined_data[combined_data.iloc[:, -1] != 0]
    
    # Prepare background data
    X_background = background_data.iloc[:, :-1].values
    y_background = np.zeros(len(X_background))  # All 0s for background
    
    # Split background data into train and test sets
    X_train, X_test_bg, y_train, y_test_bg = train_test_split(
        X_background, y_background, test_size=test_size, random_state=random_state
    )
    
    # Prepare drone data
    X_drone = drone_data.iloc[:, :-1].values
    y_drone = np.ones(len(X_drone))  # All 1s for drones (anomalies)
    
    # Combine some drone data with test set
    X_test = np.vstack((X_test_bg, X_drone))
    y_test = np.hstack((y_test_bg, y_drone))
    
    # Shuffle the test set
    test_shuffle = np.random.permutation(len(y_test))
    X_test = X_test[test_shuffle]
    y_test = y_test[test_shuffle]
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler