# -*- coding: utf-8 -*-

import numpy as np

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
    padded = np.r_[0, preds, 0]
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