# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


"""


import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
from wettbewerb import get_6montages 
import mne
from scipy import signal as sig
import ruptures as rpt
from scipy.signal import welch
import random
import joblib
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


def fix_channels(channels: list, data: np.ndarray, expected_channels: set) -> np.ndarray:
    """
    Stellt sicher, dass alle erwarteten EEG-Kanäle vorhanden sind, indem fehlende Kanäle durch Duplikate ergänzt werden.

    Wenn genau die Kanäle "Fz" und "Pz" fehlen, aber "Cz" vorhanden ist, wird "Cz" dupliziert, um diese zu ersetzen.
    In allen anderen Fällen werden zufällige vorhandene Kanäle so oft dupliziert, bis die erwartete Anzahl erreicht ist.

    Parameter
    ---------
    channels : list of str
        Liste der vorhandenen Kanalnamen.
    data : np.ndarray
        EEG-Datenmatrix mit der Form (Anzahl_Kanäle, Anzahl_Samples).
    expected_channels : set of str
        Menge der erwarteten Kanalnamen.

    Rückgabewert
    ------------
    np.ndarray
        Datenmatrix mit exakt so vielen Zeilen wie erwartete Kanäle. Fehlende Kanäle werden durch Kopien ergänzt.
    """
    
    present = set(channels)
    missing = expected_channels - present

    if len(channels) == 17 and missing == {'Fz', 'Pz'} and 'Cz' in channels:
        cz_index = channels.index('Cz')
        cz_data = data[cz_index]
        data = np.vstack([data, cz_data, cz_data])
        channels += ['Cz', 'Cz']
    else:
        needed = 19 - len(channels)
        if needed > 0:
            for _ in range(needed):
                rand_idx = random.randint(0, len(channels) - 1)
                data = np.vstack([data, data[rand_idx]])
                channels.append(channels[rand_idx])

    return data

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
    freqs_cache = None
    psd_cache = {}
    
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


def remove_lone_positives(preds, min_consecutive=2):
    """
    Entfernt isolierte positive Vorhersagen (z. B. vereinzelte Einsen) aus einer binären Vorhersagesequenz.
    
    Parameter
    ---------
    preds : array_like
        Binäres Array, das eine Vorhersagesequenz repräsentiert.
    min_consecutive : int, optional
        Minimale Anzahl aufeinanderfolgender Einsen, damit eine Sequenz als gültig betrachtet wird (Standard: 2).

    Rückgabewert
    ------------
    np.ndarray
        Gefilterte Vorhersagesequenz gleicher Länge, bei der nur zusammenhängende Einsen mit ausreichender Länge beibehalten wurden.
    """
    
    preds = np.asarray(preds).astype(int)
    # Pad with 0 at both ends to detect runs at boundaries
    padded = np.r_[0, preds, 0]
    # diff gives +1 at start of 1-run, -1 at end
    diff = np.diff(padded)
    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0]
    run_lengths = run_ends - run_starts

    cleaned = np.zeros_like(preds)
    for start, length in zip(run_starts, run_lengths):
        if length >= min_consecutive:
            cleaned[start:start+length] = 1
    return cleaned

def get_onset_of_longest_sequence(cleaned):
    """
    Gibt den Startindex der längsten zusammenhängenden Einsen-Sequenz in einem binären Array zurück.

    Parameter
    ---------
    cleaned : array_like
        Binäres Array : eine bereinigte Vorhersagesequenz.

    Rückgabewert
    ------------
    int oder None
        Der Startindex der längsten Einsen-Sequenz. Falls keine Eins vorhanden ist, wird `None` zurückgegeben.
    """
    max_len = 0
    max_start = None
    current_len = 0
    current_start = None

    for i, val in enumerate(cleaned):
        if val == 1:
            if current_len == 0:
                current_start = i
            current_len += 1
            if current_len > max_len:
                max_len = current_len
                max_start = current_start
        else:
            current_len = 0
    return max_start

def create_montages_data(channels, montages_names, data):
    """
    Erstellt bipolare Montagen aus EEG-Daten basierend auf einer Liste von Kanalpaaren.

    Parameter
    ---------
    channels : list of str
        Liste der Kanalnamen im Originalsignal.
    montages_names : dict
        Dictionary mit Montage-Namen als Schlüssel und Tupeln der Form (Kanal_A, Kanal_B) als Werte.
    data : np.ndarray
        EEG-Datenmatrix der Form (n_kanäle, n_samples).

    Rückgabewert
    ------------
    np.ndarray
        Array mit berechneten bipolaren Montagen der Form (n_montagen, n_samples).
    """
    channels = [channel.strip() for channel in channels]
    channel_indices = {ch: i for i, ch in enumerate(channels)}
    num_montages = len(montages_names)
    num_samples = data.shape[1]
    montages_data = np.zeros((num_montages, num_samples))
    for i ,(name, (a, b)) in enumerate(montages_names.items()):
        if a in channels and b in channels:
            idx_a = channel_indices[a]
            idx_b = channel_indices[b]
            diff_signal = data[idx_a] - data[idx_b]
            montages_data[i]= diff_signal
    return montages_data

def predict_sequence(seq_data, fs, model, window_size, step_size, min_consecutive):
    """
    Führt eine sequentielle Vorhersage von Anfällen in einem EEG-Signal durch.

    Das EEG-Signal wird in überlappende Fenster unterteilt, für die jeweils Merkmale extrahiert
    und einem Klassifikationsmodell übergeben werden. Einzelne oder kurze positive Vorhersagen
    werden entfernt. Bei detektiertem Anfall wird der Startzeitpunkt der längsten
    vorhergesagten Anfall-Sequenz zurückgegeben.

    Parameter
    ---------
    seq_data : np.ndarray
        EEG-Datenmatrix der Form (n_montages, n_samples).
    fs : float
        Abtastrate des Signals in Hertz.
    model : sklearn-ähnliches Modell
        Ein trainiertes Klassifikationsmodell mit `.predict()`-Methode.

    Rückgabewerte
    -------------
    seizure_detected : bool
        True, wenn ein Anfall vorhergesagt wurde; sonst False.
    predicted_onset : int oder float
        Index des vorhergesagten Anfallsbeginns (Fensterindex, nicht Zeit in Sekunden). 0.0, falls kein Anfall erkannt wurde.
    """
    
    n_chan, n_samples = seq_data.shape
    win_len_samples = int(window_size * fs)
    step_samples = int(step_size * fs)
    seizure_detected = False
    predicted_onset = 0.0
    all_preds = []
    windows= []
    for start in range(0, n_samples - win_len_samples + 1, step_samples):
        end = start + win_len_samples
        window = seq_data[:, start:end]
        windows.append(window)
    feats = [extract_window_features(w, fs) for w in windows]  # list of length n_windows, each of length n_features
    X = np.vstack(feats)   # shape (n_windows, n_features)
    all_preds = model.predict(X)  # no reshape needed
    
    all_preds = remove_lone_positives(all_preds, min_consecutive)
    if 1 in all_preds:
        predicted_onset = get_onset_of_longest_sequence(all_preds)
        seizure_detected = True
    return seizure_detected, predicted_onset

def predict_labels(channels : List[str], data : np.ndarray, fs : float, reference_system: str, model_name : str='model.json') -> Dict[str,Any]:
    '''
    Parameters
    ----------
    channels : List[str]
        Namen der übergebenen Kanäle
    data : ndarray
        EEG-Signale der angegebenen Kanäle
    fs : float
        Sampling-Frequenz der Signale.
    reference_system :  str
        Welches Referenzsystem wurde benutzt, "Bezugselektrode", nicht garantiert korrekt!
    model_name : str
        Name eures Models,das ihr beispielsweise bei Abgabe genannt habt. 
        Kann verwendet werden um korrektes Model aus Ordner zu laden
    Returns
    -------
    prediction : Dict[str,Any]
        enthält Vorhersage, ob Anfall vorhanden und wenn ja wo (Onset+Offset)
    '''

    # Initialisiere Return (Ergebnisse)
    seizure_present = True # gibt an ob ein Anfall vorliegt
    seizure_confidence = 0.5 # gibt die Unsicherheit des Modells an (optional)
    onset = 4.2   # gibt den Beginn des Anfalls an (in Sekunden)
    onset_confidence = 0.99 # gibt die Unsicherheit bezüglich des Beginns an (optional)
    offset = 999999  # gibt das Ende des Anfalls an (optional)
    offset_confidence = 0   # gibt die Unsicherheit bezüglich des Endes an (optional)
    # Preprocessing Parameters
    expected_channels = {'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'}
    montages_names = {
        'Fp1-F7': ('Fp1', 'F7'),
        'F7-T3': ('F7', 'T3'),
        'T3-T5': ('T3', 'T5'),
        'T5-O1': ('T5', 'O1'),
        'T3-C3': ('T3', 'C3'),
        'C3-Cz': ('C3', 'Cz'),
        'Fp1-F3': ('Fp1', 'F3'),
        'F3-C3': ('F3', 'C3'),
        'C3-P3': ('C3', 'P3'),
        'P3-O1': ('P3', 'O1'),
        'Fp2-F8': ('Fp2', 'F8'),
        'F8-T4': ('F8', 'T4'),
        'T4-T6': ('T4', 'T6'),
        'T6-O2': ('T6', 'O2'),
        'C4-T4': ('C4', 'T4'),
        'Cz-C4': ('Cz', 'C4'),
        'Fp2-F4': ('Fp2', 'F4'),
        'F4-C4': ('F4', 'C4'),
        'C4-P4': ('C4', 'P4'),
        'P4-O2': ('P4', 'O2'),
    }
    # Fenstergröße Parameters
    window_size = 4
    step_size = 1
    # Postprocessing Parameters
    min_consecutive = 8

    # Vortrainiertes Modell laden
    model = joblib.load(model_name)

    data = fix_channels(channels, data, expected_channels)
    montages_data = create_montages_data(channels, montages_names, data)
    try:
        seizure_present, onset = predict_sequence(montages_data, fs, model, window_size, step_size, min_consecutive)

        if not seizure_present: onset = 0.0

    except Exception as e:
        print(f"Failed on: {e}")

    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}

    return prediction
