import numpy as np
import pandas as pd

def generate_emg_signal(n=5000):
    np.random.seed(42)
    time = np.arange(n)
    emg = np.abs(np.random.normal(0, 1, n))
    return pd.DataFrame({"time": time, "emg": emg})

def preprocess_signal(df, window_len=200):
    df['emg_norm'] = (df['emg'] - df['emg'].min()) / (df['emg'].max() - df['emg'].min())

    # inject window-level anomalies
    df['emg_anomaly'] = df['emg_norm'].copy()
    num_windows = len(df) // window_len
    anomalous_windows = np.random.choice(num_windows, size=5, replace=False)

    for w in anomalous_windows:
        start = w * window_len
        end = start + window_len
        df.loc[start:end, 'emg_anomaly'] *= 6

    return df, anomalous_windows

def extract_features(df, window_len=200):
    windows = []
    for i in range(0, len(df) - window_len, window_len):
        windows.append(df['emg_anomaly'].iloc[i:i+window_len].values)

    features = []
    for w in windows:
        rms = np.sqrt(np.mean(w**2))
        features.append(rms)

    return np.array(features)