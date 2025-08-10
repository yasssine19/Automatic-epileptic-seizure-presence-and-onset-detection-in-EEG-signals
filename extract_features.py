# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis

bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 35),
    "gamma": (35, 70)
}

def petrosian_fd(signal):
    """
    Berechnet die Petrosian Fractal Dimension (PFD) eines eindimensionalen Signals.
    
    Parameter
    ---------
    signal : array_like
        Eindimensionales Array oder Liste mit dem Signalverlauf.

    Rückgabewert
    ------------
    float
        Die Petrosian Fractal Dimension des Signals. Gibt 0.0 zurück, falls das Signal leer ist
        oder keine Null-Durchgänge enthält.
    """
    diff = np.diff(signal)
    zero_crossings = np.sum(np.diff(np.sign(diff)) != 0)
    n = len(signal)
    if n == 0:
        return 0.0
    return np.log10(n) / (np.log10(n) + np.log10(n/(n + 0.4 * zero_crossings))) if zero_crossings != 0 else 0.0

def monotonicity(signal):
    """
    Zählt die Anzahl der Richtungswechsel im Signal (Monotonieverstöße).
    
    Parameter
    ---------
    signal : array_like
        Eindimensionales Signalarray.

    Rückgabewert
    ------------
    int
        Anzahl der Vorzeichenwechsel (Richtungswechsel) im Signal.
    """
    return np.sum(np.diff(np.sign(signal)) != 0)

def extract_window_features(window_data, fs):
    """
    Extrahiert statistische und spektrale Merkmale aus einem EEG-Zeitfenster.

    Für jede Montage im EEG-Zeitfenster werden folgende Merkmale berechnet:
    - Statistische Werte: Mittelwert, Varianz, Schiefe, Kurtosis, Min, Max, IQR
    - Hjorth-Parameter: Mobilität und Komplexität
    - Petrosian Fractal Dimension (PFD)
    - Leistung in EEG-Frequenzbändern: Delta, Theta, Alpha, Beta, Gamma
    - Leistungsverhältnisse pro Band
    - Spektrales Zentrum (Spectral Centroid)
    - Monotonieverletzungen (Zero-Crossings)

    Parameter
    ---------
    window_data : np.ndarray
        2D-Array mit Form (n_montagee, n_samples).
    fs : float
        Abtastrate in Hz.

    Rückgabewert
    ------------
    list of float
        Eine flache Liste aller extrahierten Merkmale, montageweise aneinandergereiht.
    """
    
    n_chan, n_samples = window_data.shape
    features = []
    
    freqs, psd_all = welch(window_data, fs=fs, axis=1, nperseg=min(256, n_samples))
    means = np.mean(window_data, axis=1)
    variations = np.var(window_data, axis=1)
    stds = np.std(window_data, axis=1)
    mins = np.min(window_data, axis=1)
    maxs = np.max(window_data, axis=1)
    band_masks = {band: (freqs >= low_f) & (freqs < high_f) for band, (low_f, high_f) in bands.items()}
    kurts = kurtosis(window_data, axis=1, fisher=True)  # fisher=True for excess kurtosis
    q75 = np.percentile(window_data, 75, axis=1)
    q25 = np.percentile(window_data, 25, axis=1)
    iqrs = q75 - q25
    
    first_deriv = np.diff(window_data, axis=1)
    second_deriv = np.diff(window_data, n=2, axis=1)

    var0 = np.var(window_data, axis=1)
    var1 = np.var(first_deriv, axis=1)
    var2 = np.var(second_deriv, axis=1)

    mobilities = np.sqrt(np.divide(var1, var0, out=np.zeros_like(var1), where=var0!=0))
    complexities = np.sqrt(np.divide(var2, var1, out=np.zeros_like(var2), where=var1!=0)) / np.where(mobilities == 0, 1, mobilities)

    for ch in range(n_chan):
        sig = window_data[ch, :]
        
        # Statistical features
        mean = means[ch]
        var = variations[ch]
        std = stds[ch]
        sig_min = mins[ch]
        sig_max = maxs[ch]
        skew = 0.0 if std == 0 else np.mean(((sig-mean)/std)**3)  # skewness
        kurt = kurts[ch]
        iqr = iqrs[ch]
        
        # Hjorth parameters
        complexity = complexities[ch]
        mobility = mobilities[ch]
        
        # Petrosian fractal dimension
        pfd = petrosian_fd(sig)
        
        # Frequency domain features (band powers and ratios)
        psd = psd_all[ch, :]
        total_power = np.trapezoid(psd, freqs)
        band_powers = {}
        for band, (low_f, high_f) in bands.items():
            band_mask = band_masks[band]
            band_power = np.trapezoid(psd[band_mask], freqs[band_mask]) if band_mask.any() else 0.0
            band_powers[band] = band_power
            
        # Power ratios and spectral centroid
        ratios = {}
        for band in bands:
            ratios[band] = band_powers[band] / total_power if total_power > 0 else 0.0
            
        # Spectral centroid (frequency weighted by power)
        spectral_centroid = (np.sum(freqs * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0.0
        
        # Monotonicity (zero-crossings count)
        mono = monotonicity(sig)
        
        ch_features = [mean, var, skew, kurt, iqr, sig_min, sig_max,
                       mobility, complexity, pfd,
                       band_powers['delta'], band_powers['theta'], band_powers['alpha'],
                       band_powers['beta'], band_powers['gamma'],
                       ratios['delta'], ratios['theta'], ratios['alpha'],
                       ratios['beta'], ratios['gamma'],
                       spectral_centroid, mono]
        features.extend(ch_features)
    return features

