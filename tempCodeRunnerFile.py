def remove_noise(signal, threshold):
    return signal[np.abs(signal) > threshold]