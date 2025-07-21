# -*- coding: utf-8 -*-
"""
Beispiel Code und  Spielwiese

"""


import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from wettbewerb import load_references, get_3montages
import mne
from scipy import signal as sig
import ruptures as rpt
import json


### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

training_folder  = "../shared_data/training_mini"


ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder) # Importiere EEG-Dateien, zugehörige Kanalbenennung, Sampling-Frequenz (Hz) und Name (meist fs=256 Hz), sowie Referenzsystem



# Seizure Detektion (Der Beispielcode speichert hier ein Modell)
# Initialisiere Datenarrays
feature = []
label = []

for i,_id in enumerate(ids):
    _fs = sampling_frequencies[i]
    _eeg_signals = data[i]
    _eeg_label = eeg_labels[i]
    label.append(_eeg_label[0])
    # Berechne Montage
    _montage, _montage_data, _is_missing = get_3montages(channels[i], _eeg_signals)
    signal_std = np.zeros(len(_montage))
    for j, signal_name in enumerate(_montage):
        # Ziehe erste Montage des EEG
        signal = _montage_data[j]
        # Wende Notch-Filter an um Netzfrequenz zu dämpfen
        signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
        # Wende Bandpassfilter zwischen 0.5Hz und 70Hz um Rauschen aus dem Signal zu filtern
        signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
        

        signal_std[j] = np.std(signal_filter)

    # Nur der Kanal mit der maximalen Standardabweichung wird berücksichtigt
    signal_std_max = signal_std.max()
    feature.append(signal_std_max)

X = np.array(feature)
Y = np.array(label)    
best_f1 = 0
th_opt = 0
for th in np.arange(X.min(),X.max(),(X.max()-X.min())/1e5):
    pred = X>th
    TP = np.sum((pred==Y) & (Y==1))
    FP = np.sum((pred==1) & (Y==0))
    FN = np.sum((pred==0) & (Y==1))
    F1 = 2*TP/(2*TP+FP+FN)  
    if F1 >best_f1:
        th_opt = th
        best_f1 = F1
print('Optimaler Threshold ist', th_opt,' bei F1 auf Trainingsdaten von',best_f1)

# Speichere Modell
model_params = {'std_thresh':th_opt}
with open('model.json', 'w', encoding='utf-8') as f:
    json.dump(model_params, f, ensure_ascii=False, indent=4)
    print('Seizure Detektionsmodell wurde gespeichert!')
        


# Onset Detektion (Der Beispielcode speichert hier kein Modell, da keine Parameter gelernt werden)
# Initialisiere Datenarrays
onset_list_predict = []
onset_list = []
seizure_id_list = []

for i,_id in enumerate(ids):
    _fs = sampling_frequencies[i]
    _eeg_signals = data[i]
    _eeg_label = eeg_labels[i]
    if _eeg_label[0]:
        onset_list.append(_eeg_label[1])
        seizure_id_list.append(_id)
        # Berechne Montage
        _montage, _montage_data, _is_missing = get_3montages(channels[i], _eeg_signals)
        for j, signal_name in enumerate(_montage):
            # Ziehe erste Montage des EEG
            signal = _montage_data[j]
            # Wende Notch-Filter an um Netzfrequenz zu dämpfen
            signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
            # Wende Bandpassfilter zwischen 0.5Hz und 70Hz um Rauschen aus dem Signal zu filtern
            signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
            
            # Berechne short time fourier transformation des Signal: signal_filtered = filtered signal of channel, fs = sampling frequency, nperseg = length of each segment
            # Output f= array of sample frequencies, t = array of segment times, Zxx = STFT of signal
            f, t, Zxx = sig.stft(signal_filter, _fs, nperseg=_fs * 3)
            # Berechne Schrittweite der Frequenz
            df = f[1] - f[0]
            # Berechne Engergie (Betrag) basierend auf Real- und Imaginärteil der STFT
            E_Zxx = np.sum(Zxx.real ** 2 + Zxx.imag ** 2, axis=0) * df

    

            # Erstelle neues Array in der ersten Iteration pro Patient
            if j == 0:
                # Initilisiere Array mit Energiesignal des ersten Kanals
                E_array = np.array(E_Zxx)
            else:
                # Füge neues Energiesignal zu vorhandenen Kanälen hinzu (stack it)
                E_array = np.vstack((E_array, np.array(E_Zxx)))
                

        # Berechne Gesamtenergie aller Kanäle für jeden Zeitppunkt
        E_total = np.sum(E_array, axis=0)
        # Berechne Stelle der maximalen Energie
        max_index = E_total.argmax()

        # Berechne "changepoints" der Gesamtenergie
        # Falls Maximum am Anfang des Signals ist muss der Onset ebenfalls am Anfang sein und wir können keinen "changepoint" berechnen
        if max_index == 0:
            onset_list_predict.append(0.0)
        else:
            # Berechne "changepoint" mit dem ruptures package
            # Setup für  "linearly penalized segmentation method" zur Detektion von changepoints im Signal mi rbf cost function
            algo = rpt.Pelt(model="rbf").fit(E_total)
            # Berechne sortierte Liste der changepoints, pen = penalty value
            result = algo.predict(pen=10)
            #Indices sind ums 1 geshiftet
            result1 = np.asarray(result) - 1
            # Selektiere changepoints vor Maximum
            result_red = result1[result1 < max_index]
            # Falls es mindestens einen changepoint gibt nehmen wir den nächsten zum Maximum
            if len(result_red)<1:
                # Falls keine changepoint gefunden wurde raten wir, dass er "nahe" am Maximum ist
                print('No changepoint, taking maximum')
                onset_index = max_index
            else:
                # Der changepoint entspricht gerade dem Onset 
                onset_index = result_red[-1]
            # Füge Onset zur Liste der Onsets hinzu
            onset_list_predict.append(t[onset_index])

# Compute absolute error between compute seizure onset and real onset based on doctor annotations
prediction_error = np.abs(np.asarray(onset_list_predict) - np.asarray(onset_list))
print('Mittlerer Onset Prädiktionsfehler Training:', np.mean(prediction_error))

# Plot error per patient
plt.figure(1)
plt.scatter(np.arange(1, len(prediction_error)+1),prediction_error)
#plt.hlines(10, 0, len(prediction_error)+1, colors='red')
plt.ylabel('Error in s')
plt.xlabel('Patients')
plt.show()