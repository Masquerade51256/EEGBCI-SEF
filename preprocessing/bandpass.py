import scipy    

def bandpass_filter(data, low_freq, high_freq, sfreq, order=4):
        """
        Bandpass filter using a Butterworth filter.
        Args:
            data: Input data, shape (..., n_timepoints). This function will filter along the last axis (time axis).
            low_freq: Low cutoff frequency of the passband.
            high_freq: High cutoff frequency of the passband.
            sfreq: Sampling frequency.
            order: Order of the filter.
        Returns:
            filtered_data: Filtered data.
        """
        # Design Butterworth bandpass filter
        nyquist = sfreq / 2.0
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        
        # Apply filter along the last axis (axis=-1)
        filtered_data = scipy.signal.filtfilt(b, a, data, axis=-1)
        return filtered_data