# -*- coding: utf-8 -*-

import os
import glob
import joblib
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
from extract_features import extract_window_features
from preprocessing import fix_channels, create_montages_data
from postprocessing import remove_lone_positives, get_onset_of_longest_sequence

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

def run_test_predictions(test_records, model, out_dir, min_consecutive=8):
    """
    Führt Vorhersagen auf allen Test-Records durch und gibt eine Liste
    von (record_id, pred_label, pred_onset, true_label, true_onset) zurück.

    Für jeden Datensatz:
    1. Lädt die zugehörigen Features (`*_feats.npy`) und Labels (`*_labels.npy`).
    2. Überprüft, ob die Feature-Form der Erwartung entspricht
       (in diesem Fall: 22 Features pro Montage → 440 Features bei 20 Montages).
    3. Führt Vorhersage mit dem übergebenen Modell durch.
    4. Postprocessing:
       - Entfernt isolierte Positive, die nicht in einer Sequenz von mindestens
         `min_consecutive` Fenstern vorkommen (`remove_lone_positives`).
       - Falls mindestens ein positives Fenster verbleibt:
         * Bestimmt den Onset aus der längsten Sequenz (`get_onset_of_longest_sequence`).
       - Andernfalls: Onset = 0.0.
    5. Speichert für jeden Datensatz:
       (record_id, vorhergesagte Präsenz, vorhergesagter Onset, Ground-Truth-Präsenz, Ground-Truth-Onset).

    Parameter
    ---------
    test_records : pandas.DataFrame
        DataFrame mit mindestens den Spalten:
        - "record_id": eindeutiger String/ID für den Record
        - "seizure_label": 0 oder 1
        - "onset": Ground-Truth-Onset in Sekunden
    model : object
        Modell mit einer `.predict(X)`-Methode, z. B. `ClassifierModel` oder Sklearn-Modell.
    out_dir : str
        Pfad zu dem Ordner, der die vorverarbeiteten Test-Feature- und Label-Dateien enthält.
    min_consecutive : int, default=8
        Mindestanzahl aufeinanderfolgender positiver Fenster, um als Seizure zu gelten.

    Rückgabewert
    ------------
    list of tuple
        Liste von Tupeln:
        (record_id, pred_label, pred_onset, true_label, true_onset)
    """
    test_results = []
    rows = test_records.to_dict(orient="records")
    for row in tqdm(rows, desc='Predicting'):
        record_id = row["record_id"]
        true_label = row["seizure_label"]
        true_onset = row["onset"]
        seizure_detected = False

        # Get all feature and label file paths
        feature_file = sorted(glob.glob(os.path.join(out_dir, f"{record_id}_feats.npy")))
        label_file   = sorted(glob.glob(os.path.join(out_dir, f"{record_id}_labels.npy")))

        if not feature_file or not label_file:
            continue

        feats = np.load(feature_file[0])
        if feats.shape[1] != 440:
            continue  # skip if feature shape is not as expected (In our case we have 22 features per montage)
        labels = np.load(label_file[0])

        y_preds = model.predict(feats)
        y_preds = remove_lone_positives(y_preds, min_consecutive=min_consecutive)
        if 1 in y_preds:
            pred_onset = get_onset_of_longest_sequence(y_preds)
            seizure_detected = True
        else:
            pred_onset = 0.0

        pred_label = 1 if seizure_detected else 0

        test_results.append((record_id, pred_label, pred_onset, true_label, true_onset))
    return test_results