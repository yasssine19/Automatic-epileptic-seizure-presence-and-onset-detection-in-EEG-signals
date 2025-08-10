# -*- coding: utf-8 -*-

import os
import gc
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io as sio
import concurrent.futures
from extract_features import extract_window_features

data_folder = "../shared_data/training"

# EEG-Frequenzbänder (Hz) – passend zu eurer Feature-Extraktion
bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 35),
    "gamma": (35, 70)
}

# 20 Bipolar-Montagen (International 10–20 System, Beispiel: 'Fp1-F3')
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

def balanced_patient_split(ref_df, eligible_patients, test_size=0.2, random_state=40):
    """
    Teilt Patienten in Train/Test, versucht dabei eine grob ausgeglichene
    Verteilung der Klassen (0/1) über beide Splits zu erreichen.

    Strategie
    ---------
    - Shuffelt die Liste `eligible_patients` (reproduzierbar per `random_state`).
    - Iteriert über Patienten und füllt zunächst den Testsplit bis `test_size`.
      Für jeden Patienten werden die Labelzählungen in den jeweiligen Split
      addiert. (Heuristik: einfaches Befüllen macht in vielen Settings schon
      einen guten Job, weil auf Patientenebene gemischt wird.)
    - Der Rest geht in den Trainingssplit.

    Parameter
    ---------
    ref_df : pd.DataFrame
        Referenztabelle mit mindestens den Spalten:
        - "patient_id" : ID des Patienten
        - "seizure_label" : 0/1 Label pro Sample/Record
    eligible_patients : List[str]
        Liste der Patienten-IDs, die überhaupt berücksichtigt werden sollen
        (z. B. nur Patienten mit mindestens einem Sample jeder Klasse).
    test_size : float, default=0.2
        Zielanteil an Patienten im Testsplit (zwischen 0 und 1).
    random_state : int, default=40
        Seed für die Reproduzierbarkeit.

    Rückgabewert
    ------------
    (train_records, test_records) : Tuple[pd.DataFrame, pd.DataFrame]
        Teilmengen von `ref_df` für die jeweiligen Splits.
    """
    np.random.seed(random_state)
    np.random.shuffle(eligible_patients)

    train_ids, test_ids = [], []
    train_counts, test_counts = {0: 0, 1: 0}, {0: 0, 1: 0}

    for pid in eligible_patients:
        patient_records = ref_df[ref_df["patient_id"] == pid]
        label_counts = patient_records["seizure_label"].value_counts().to_dict()

        train_diff = abs(train_counts[0] - train_counts[1])
        test_diff = abs(test_counts[0] - test_counts[1])

        if len(test_ids) < test_size * len(eligible_patients):
            test_ids.append(pid)
            for label, count in label_counts.items():
                test_counts[label] += count
        else:
            train_ids.append(pid)
            for label, count in label_counts.items():
                train_counts[label] += count

    train_records = ref_df[ref_df["patient_id"].isin(train_ids)].reset_index(drop=True)
    test_records  = ref_df[ref_df["patient_id"].isin(test_ids)].reset_index(drop=True)

    return train_records, test_records

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

def process_record(row, output_dir):
    """
    Extrahiert Features/Labels für einen einzelnen Record und speichert sie als .npy.

    Erwartet, dass ein globaler `data_folder` existiert, in dem die .mat-Dateien
    unter `<data_folder>/<record_id>.mat` liegen.

    Parameter
    ---------
    row : dict
        Ein Zeilen-Dict mit mindestens den Keys:
        - "record_id": str – Basisname der .mat-Datei
        - "seizure_label": int (0/1)
        - "onset": float – Startzeit der Seizure in Sekunden
        - "offset": float – Endzeit in Sekunden (<=0 bedeutet: bis Ende der Aufnahme)
    output_dir : str
        Ausgabeverzeichnis, in dem *_feats.npy und *_labels.npy abgelegt werden.

    Rückgabewert
    ------------
    (features, labels) : Tuple[list, list]
        Listen der extrahierten Featurevektoren und zugehörigen Labels.
        (Werden unabhängig davon auf Platte gespeichert.)
    """
    record_id = row["record_id"]
    label = row["seizure_label"]
    onset = row["onset"]
    offset = row["offset"]

    os.makedirs(output_dir, exist_ok=True)

    # Konfigurierbar
    window_size = 4
    step_size = 1
    
    try:
        mat = sio.loadmat(f"{data_folder}/{record_id}.mat")
        signals = mat.get('data') if mat.get('data') is not None else mat['val']
        signals = signals.astype(np.float64)
        channels = mat.get("channels", None)
        montages_data = create_montages_data(channels, montages_names, signals)
                
        fs = float(mat['fs'].item()) if 'fs' in mat else 256.0
        n_chan, n_samples = signals.shape

        win_len_samples = int(window_size * fs)
        step_samples = int(step_size * fs)

        features = []
        labels = []

        for start in range(0, n_samples - win_len_samples + 1, step_samples):
            end = start + win_len_samples
            window = montages_data[:, start:end]
            feat = extract_window_features(window, fs)
            features.append(feat)

            if label == 1:
                seizure_end = offset if offset > 0 else n_samples / fs
                window_start_sec = start / fs
                window_end_sec = end / fs
                seizure_in_window = (window_end_sec >= onset) and (window_start_sec <= seizure_end)
                labels.append(1 if seizure_in_window else 0)
            else:
                labels.append(0)

        np.save(os.path.join(output_dir, f"{record_id}_feats.npy"), features)
        np.save(os.path.join(output_dir, f"{record_id}_labels.npy"), labels)

        del mat, signals, features, labels, window, feat, montages_data
        gc.collect()
    
    except Exception as e:
        print(f"Failed to process {record_id}: {e}")
        return [], []
    
def process_record_wrapper(args):
    """
    Thin Wrapper, um `ProcessPoolExecutor.map` mit Tupelargumenten zu füttern.
    """
    return process_record(*args)

def process_reference_data(data_folder):
    """
    Liest REFERENCE.csv ein und konstruiert patientenweise Metadaten.

    Erwartete CSV-Struktur (ohne Header):
        record_id, seizure_label, onset, offset

    Ergänzt:
        - patient_id = record_id.split("_")[0]

    Parameter
    ---------
    data_folder : str
        Verzeichnis, das die `REFERENCE.csv` enthält.

    Rückgabewert
    ------------
    (ref_df, eligible_patients) : Tuple[pd.DataFrame, List[str]]
        - ref_df : DataFrame mit Spalten ["record_id","seizure_label","onset","offset","patient_id"]
        - eligible_patients : Patienten-IDs mit mind. einem Sample Klasse 0 *und* 1
    """
    ref_df = pd.read_csv(f"{data_folder}/REFERENCE.csv", header=None)
    ref_df.columns = ["record_id", "seizure_label", "onset", "offset"]
    ref_df["seizure_label"] = ref_df["seizure_label"].astype(int)
    ref_df["onset"] = ref_df["onset"].astype(float)
    ref_df["offset"] = ref_df["offset"].astype(float)
    ref_df["patient_id"] = ref_df["record_id"].apply(lambda x: x.split("_")[0])

    # Group records by patient and seizure label
    grouped = ref_df.groupby(["patient_id", "seizure_label"]).size().unstack(fill_value=0)

    # Only keep patients with at least one sample of each class
    eligible_patients = grouped[(grouped[0] > 0) & (grouped[1] > 0)].index.tolist()

    return ref_df,eligible_patients

def process_records_parallel(records,output_dir):
    """
    Verarbeitet mehrere Records parallel: Feature- und Label-Dateien erzeugen.

    Erwartet globales `data_folder`, auf das `process_record` zugreift.

    Parameter
    ---------
    records : pd.DataFrame
        DataFrame mit Zeilen, die mindestens die Keys für `process_record` enthalten.
    output_dir : str
        Zielordner für die abgelegten .npy-Dateien.
    """
     # process recors in Parallel
    rows = records.to_dict(orient="records")  # list of dicts
    args_list = [(row, output_dir) for row in rows]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_record_wrapper, args_list),
                            total=len(rows),
                            desc="Extracting features"))
        
def load_data(feature_files,label_files):
    """
    Lädt .npy-Feature/Label-Dateien in flache Listen X, y für das Training.

    Es werden nur Records berücksichtigt, die mindestens ein positives Fenster
    enthalten, und nur Fenster mit vollständiger Feature-Länge (440).

    Parameter
    ---------
    feature_files : List[str]
        Pfade zu `*_feats.npy`-Dateien.
    label_files : List[str]
        Pfade zu `*_labels.npy`-Dateien (gleiche Reihenfolge wie `feature_files`).

    Rückgabewert
    ------------
    (X, y) : Tuple[List[List[float]], List[int]]
        - X : Liste der Featurevektoren (Fenster)
        - y : korrespondierende Labels (0/1)
    """
    X = []
    y = []
    for feat_file, label_file in zip(feature_files, label_files):
        feats = np.load(feat_file)
        labels = np.load(label_file)
        seizure_pres = 1 in labels
        if feats.ndim != 2:
            print(f"Skipping {os.path.basename(feat_file)} due to shape {feats.shape}")
            continue
        if seizure_pres:
            for label, feat in zip(labels, feats):
                if len(feat) != 440:
                    break
                else:
                    y.append(label)
                    X.append(feat)
    
    return X,y