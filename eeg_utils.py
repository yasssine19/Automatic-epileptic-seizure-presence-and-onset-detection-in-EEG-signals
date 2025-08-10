# -*- coding: utf-8 -*-
"""
Diese Datei sollte nicht verändert werden und wird von uns gestellt und zurückgesetzt.

Funktionen zum Laden und Speichern der Dateien
"""
__author__ = "Maurice Rohr und Dirk Schweickard"

from typing import List, Tuple, Dict, Any
import csv
import scipy.io as sio
import numpy as np
import os


def load_references(folder: str = '../training', idx: int = 0) -> Tuple[List[str], List[List[str]],
                                                          List[np.ndarray],  List[float],
                                                          List[str], List[Tuple[bool,float,float]]]:
    """
    Liest Referenzdaten aus .mat (Messdaten) und .csv (Label) Dateien ein.
    Parameters
    ----------
    folder : str, optional
        Ort der Trainingsdaten. Default Wert '../training'.
    idx : int, optinal
        Index, ab dem das Laden der Daten starten soll.
        z.B. idx=10 bedeutet es werden die Datenpunkte 10 bis 109 (einschließlich) geladen
        Falls weniger als 100 Dateien ab Index übrig, werden nur diese geladen.

    Returns
    -------
    ids : List[str]
        Liste von ID der Aufnahmen
    channels : List[List[str]]
        Liste der vorhandenen Kanäle per Aufnahme
    data :  List[ndarray]
        Liste der Daten pro Aufnahme
    sampling_frequencies : List[float]
        Liste der Sampling-Frequenzen.
    reference_systems : List[str]
        Liste der Referenzsysteme. "LE", "AR", "Sz" (Zusatz-Information)
    """
    
    # Initialisiere Listen ids, channels, data, sampling_frequencies, refernece_systems und eeg_labels
    ids: List[str] = []
    channels: List[List[str]] = []
    data: List[np.ndarray] = []
    sampling_frequencies: List[float] = []
    reference_systems: List[str] = []
    eeg_labels: List[Tuple[bool,float,float]] = []
    
    # Erzeuge Datensatz aus Ordner und fülle Listen mit Daten
    dataset = EEGDataset(folder)
    data_to_load = 100  # lade bis zu 100 Aufnahmen
    if(len(dataset)-idx < data_to_load):
        end = len(dataset)-idx
    else:
        end = data_to_load
    if(idx >= len(dataset)):
        print("Dataset is smaller than the provided idx")
        return None
    for i in range(idx, idx+end):
        ids.append(dataset[i][0])
        channels.append(dataset[i][1])
        data.append(dataset[i][2])
        sampling_frequencies.append(dataset[i][3])
        reference_systems.append(dataset[i][4])
        eeg_labels.append(dataset[i][5])
        
    # Zeige an wie viele Daten geladen wurden
    print("{}\t Dateien wurden geladen.".format(len(ids)))
    return ids, channels, data, sampling_frequencies, reference_systems, eeg_labels

class EEGDataset:
    def __init__(self,folder:str) -> None:
        """Diese Klasse stellt einen EEG Datensatz dar.
        
        Verwendung:
            Erzeuge einen neuen Datensatz (ohne alle Daten zu laden) mit
            dataset = EEGDataset("../training/")
            len(dataset) # gibt Größe des Datensatzes zurück
            dataset[0] # gibt erstes Element aus Datensatz zurück bestehend aus (id, channels, data, sampling_frequency, reference_system, eeg_label)
            it = iter(dataset) # gibt einen iterator zurück auf den Datensatz,
            next(it) # gibt nächstes Element zurück bis alle Daten einmal geholt wurden
            for item in dataset: # iteriert einmal über den gesamten Datensatz
                (id, channels, data, sampling_frequency, reference_system, eeg_label) = item
                # Berechnung

        Args:
            folder (str): Ordner in dem der Datensatz bestehend aus .mat-Dateien und einer REFERENCE.csv Datei liegt
        """
        assert isinstance(folder, str), "Parameter folder muss ein string sein aber {} gegeben".format(type(folder))
        assert os.path.exists(folder), 'Parameter folder existiert nicht!'
        # Initialisiere Listen für ids und labels
        self._folder = folder
        self._ids: List[str] = []
        self._eeg_labels: List[Tuple[bool,float,float]] = []
        # Lade references Datei
        with open(os.path.join(folder, 'REFERENCE.csv')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # Iteriere über jede Zeile
            for row in csv_reader:
                self._ids.append(row[0])
                self._eeg_labels.append((int(row[1]),float(row[2]),float(row[3])))
    
    def __len__(self):
        return len(self._ids)
    
    def __getitem__(self,idx) -> Tuple[str, List[str],
                                    np.ndarray,  float,
                                    str, Tuple[bool,float,float]]:
        #Lade Matlab-Datei
        eeg_data = sio.loadmat(os.path.join(self._folder, self._ids[idx] + '.mat'),simplify_cells=True)
        ch_names = eeg_data.get('channels')
        channels = [x.strip(' ') for x in ch_names] 
        data = eeg_data.get('data')
        sampling_frequency = eeg_data.get('fs')
        reference_system = eeg_data.get('reference_system')
        return (self._ids[idx],channels,data,sampling_frequency,reference_system,self._eeg_labels[idx])
    
    def get_labels(self):
        return self._eeg_labels
    
#predictions = {"id":id,"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
#                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
#                   "offset_confidence":offset_confidence}
def save_predictions(predictions: List[Dict[str,Any]], folder: str=None, teamID: str=None) -> None:
    """
    Funktion speichert the gegebenen predictions in eine CSV-Datei mit dem name PREDICTIONS.csv. 
    Alle Optionalen Vorherhsagen werden mit Standardwerten ersetzt.
    Parameters
    ----------
    predictions : List[Dict[str,Any]]
        Liste aus dictionaries. Jedes Dictionary enthält die Felder "id","seizure_present",
                "seizure_confidence" (optional),"onset","onset_confidence" (optional),
                "offset" (optional),"offset_confidence" (optional)
	folder : str
		Speicherort der predictions
    Returns
    -------
    None.

    """    
	# Check Parameter
    assert isinstance(predictions, list), \
        "Parameter predictions muss eine Liste sein aber {} gegeben.".format(type(predictions))
    assert len(predictions) > 0, 'Parameter predictions muss eine nicht leere Liste sein.'
    assert isinstance(predictions[0], dict), \
        "Elemente der Liste predictions muss ein Dictionary sein aber {} gegeben.".format(type(predictions[0]))
    assert "id" in predictions[0], \
        "Prädiktionen müssen eine ID besitzen, aber Key in Dictionary nicht vorhanden"
	
    if folder==None:
        file = "PREDICTIONS.csv"
    else:
        if teamID is None:
            file = os.path.join(folder, "PREDICTIONS.csv")
        else:
            file = os.path.join(folder, "PREDICTIONS"+"_"+teamID+".csv")
    # Check ob Datei schon existiert wenn ja loesche Datei
    if os.path.exists(file) and teamID is None:
        os.remove(file)

    with open(file, mode='w', newline='') as predictions_file:

        # Init CSV writer um Datei zu beschreiben
        predictions_writer = csv.writer(predictions_file, delimiter=',')
        # Iteriere über jede prediction
        header=["id","seizure_present","seizure_confidence","onset","onset_confidence","offset","offset_confidence"]
        predictions_writer.writerow(header)
        for prediction in predictions:
            _id = prediction["id"]
            _seizure_present = prediction["seizure_present"]
            _seizure_confidence = prediction.get("seizure_confidence",1.0) 
            _onset = prediction["onset"]
            _onset_confidence = prediction.get("onset_confidence",1.0) 
            _offset = prediction.get("offset",999999.0)
            _offset_confidence = prediction.get("offset_confidence",0.0)
            predictions_writer.writerow([_id,_seizure_present,_seizure_confidence,_onset,_onset_confidence,_offset,_offset_confidence])
        # Gebe Info aus wie viele labels (predictions) gespeichert werden
        print("{}\t Labels wurden geschrieben.".format(len(predictions)))
        

def get_3montages(channels: List[str], data: np.ndarray) -> Tuple[List[str],np.ndarray,bool]:
    """
    Funktion berechnet die 3 Montagen Fp1-F3, Fp2-F4, C3-P3 aus den gegebenen Ableitungen (Montagen)
    zur selben Referenzelektrode. Falls nicht alle nötigen Elektroden vorhanden sind, wird das entsprechende Signal durch 0 ersetzt. 
    ----------
    channels : List[str]
        Namen der Kanäle z.B. Fp1, Cz, C3
	data : ndarray
		Daten der Kanäle
    Returns
    -------
    montages : List[str]
        Namen der Montagen ["Fp1-F3", "Fp2-F4", "C3-P3"]
    montage_data : ndarray
        Daten der Montagen
    montage_missing : bool
        1 , falls eine oder mehr Montagen fehlt, sonst 0

    """   
    montages = []
    _,m = np.shape(data)
    montage_data = np.zeros([3,m])
    montage_missing = 0
    if '-' in channels:
        try:
            montage_data[0,:] = data[channels.index('Fp1-F3')]
            montages.append('Fp1-F3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[1,:] = data[channels.index('Fp2-F4')]
            montages.append('Fp2-F4')
        except:
            montage_missing = 1
            montages.append('error')        
        try:
            montage_data[2,:] = data[channels.index('C3-P3')]
            montages.append('C3-P3')
        except:
            montage_missing = 1
            montages.append('error')

        return (montages,montage_data,montage_missing)

    else:
        try:
            montage_data[0,:] = data[channels.index('Fp1')] - data[channels.index('F3')]
            montages.append('Fp1-F3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[1,:] = data[channels.index('Fp2')] - data[channels.index('F4')]
            montages.append('Fp2-F4')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[2,:] = data[channels.index('C3')] - data[channels.index('P3')]
            montages.append('C3-P3')
        except:
            montage_missing = 1
            montages.append('error')

        return (montages,montage_data,montage_missing)


def get_6montages(channels: List[str], data: np.ndarray) -> Tuple[List[str],np.ndarray,bool]:
    """
    Funktion berechnet die 6 Montagen Fp1-F3, Fp2-F4, C3-P3, F3-C3, F4-C4, C4-P4 aus den gegebenen Ableitungen (Montagen)
    zur selben Referenzelektrode. Falls nicht alle nötigen Elektroden vorhanden sind, wird das entsprechende Signal durch 0 ersetzt. 
    ----------
    channels : List[str]
        Namen der Kanäle z.B. Fp1, Cz, C3
	data : ndarray
		Daten der Kanäle
    Returns
    -------
    montages : List[str]
        Namen der Montagen ["Fp1-F3", "Fp2-F4", "C3-P3", "F3-C3", "F4-C4", "C4-P4"]
    montage_data : ndarray
        Daten der Montagen
    montage_missing : bool
        1 , falls eine oder mehr Montagen fehlt, sonst 0

    """  
    montages = []
    _,m = np.shape(data)
    montage_data = np.zeros([6,m])
    montage_missing = 0
    if '-' in channels:
        try:
            montage_data[0,:] = data[channels.index('Fp1-F3')]
            montages.append('Fp1-F3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[1,:] = data[channels.index('Fp2-F4')]
            montages.append('Fp2-F4')
        except:
            montage_missing = 1
            montages.append('error')        
        try:
            montage_data[2,:] = data[channels.index('C3-P3')]
            montages.append('C3-P3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[3,:] = data[channels.index('F3-C3')]
            montages.append('F3-C3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[4,:] = data[channels.index('F4-C4')]
            montages.append('F4-C4')
        except:
            montage_missing = 1
            montages.append('error')        
        try:
            montage_data[5,:] = data[channels.index('C4-P4')]
            montages.append('C4-P4')
        except:
            montage_missing = 1
            montages.append('error')

        return (montages,montage_data,montage_missing)

    else:         
        try:
            montage_data[0,:] = data[channels.index('Fp1')] - data[channels.index('F3')]
            montages.append('Fp1-F3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[1,:] = data[channels.index('Fp2')] - data[channels.index('F4')]
            montages.append('Fp2-F4')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[2,:] = data[channels.index('C3')] - data[channels.index('P3')]
            montages.append('C3-P3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[3,:] = data[channels.index('F3')] - data[channels.index('C3')]
            montages.append('F3-C3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[4,:] = data[channels.index('F4')] - data[channels.index('C4')]
            montages.append('F4-C4')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[5,:] = data[channels.index('C4')] - data[channels.index('P4')]
            montages.append('C4-P4')
        except:
            montage_missing = 1
            montages.append('error')

        return (montages,montage_data,montage_missing)
