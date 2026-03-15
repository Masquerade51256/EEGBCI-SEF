from .. import config

def apply_filter_bank(data, bank):
    # Apply filter bank to the data
    return data

def create_windows(data, window_length, window_stride):
    # Create sliding windows from the data
    windows = []
    for start in range(0, len(data) - window_length + 1, window_stride):
        window = data[start:start + window_length]
        windows.append(window)
    return windows

def preprocess_data(data):
    # Apply preprocessing steps based on config
    if config.use_filter_bank:
        data = apply_filter_bank(data, config.bank)
    else:
        data = apply_bandpass_filter(data, config.band_filter)
    if config.use_csp:
        data = apply_csp(data)
    # Sliding window
    windowed_data = create_windows(data, config.window_length, config.window_stride)
    return windowed_data