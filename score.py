import os
import sys
import pandas as pd
import numpy as np
import argparse
from typing import Tuple,List, Dict, Union


def score(test_dir : str='../test/') -> Dict[str, Union[float, List[int]]]:
    """
    Berechnet relevante Metriken des Wettbewerbs, sowie weitere Metriken
    Parameters
    ----------
    folder : str, optional
        Ort der Testdaten. Default Wert '../test'.

    Returns
    -------
    Dict, wobei die folgenden keys existieren:
        performance_metric_WS23 : float
            Metrik wird für das Ranking verwendet. F1-Score für Vorhersagen die in korrekte Umgebung fallen
        performance_metric_SS23 : float
            MAE der Onset Detektion mit Strafterm für Fehlklassifikation
        F1 : float
            F1-Score der Seizure-Klassifikation (Seizure present = postive Klasse)
        sensitivity :  float
            Sensitivität der Seizure Klassifikation
        PPV : float
            Positive Predictive Value der Seizure Klassifikation
        detection_error_onset : float
            Mittlerer Absoluter Fehler der Onset Detektion (mit oberem Limit pro Aufnahme)
        detection_error_offset : float
            Mittlerer Absoluter Fehler der Offset Detektion (mit oberem Limit pro Aufnahme)
        confusion_matrix : List[int]
            Confusion Matrix der Seizure Klassifikation [TP,FN,FP,TN]
    """
        
    if not os.path.exists("PREDICTIONS.csv"):
        sys.exit("Es gibt keine Predictions")  

    if  not os.path.exists(os.path.join(test_dir, "REFERENCE.csv")):
        sys.exit("Es gibt keine Ground Truth")  

    df_pred = pd.read_csv("PREDICTIONS.csv")   # Klassifikationen
    df_gt = pd.read_csv(os.path.join(test_dir,"REFERENCE.csv"), sep=',',header=None)  # Wahrheit

    
    ONSET_PENALTY = 60 # Sekunden, falls Anfall nicht erkannt oder größer als Penalty, wird Penalty gewertet
    FALSE_CLASSIFICATION_PENALTY = 60 # Sekunden, falls Anfall erkannt wird, obwohl keiner vorliegt, werden Strafsekunden vergeben
    INTERVAL_DELAY = 30 # Sekunden, Interval in dem der vorhergesagte Anfall um den tatsächlichen liegen darf
    
    # Stelle sicher, dass zu jedem Referenzwert eine Vorhersage existiert
    gt_seizure_present,gt_onset,gt_offset,pred_seizure_present,pred_onset,pred_offset,pred_seizure_confidence,pred_onset_confidence,pred_offset_confidence = match_predictions(df_gt,df_pred)
    # Berechne die Confusions-Matrix für das Klassifikationsproblem
    confusion_matrix = compute_confusion_matrix(gt_seizure_present,pred_seizure_present)
    # Berechne Grundlegende Metriken auf Basis von TP,FN,FP,TN
    sensitivity,PPV,F1,accuracy = compute_basic_metrics(*confusion_matrix)
    # Berechne den Fehler der Onset-Prädiktion ("Latenz")
    detection_error_onset = compute_detection_error(gt_seizure_present,pred_seizure_present,gt_onset,pred_onset,ONSET_PENALTY)
    # Berechne den Fehler der Offset-Prädiktion (optional)
    detection_error_offset = compute_detection_error(gt_seizure_present,pred_seizure_present,gt_offset,pred_offset,ONSET_PENALTY)
    # Berechne gewichtete Metrik aus Latenz und F1-Score (Sommersemester 2023) (niedriger=besser)
    performance_metric_SS23 = compute_SS23_performance_metric(detection_error_onset,confusion_matrix,FALSE_CLASSIFICATION_PENALTY)
    i_score_sensitivity,i_score_PPV,i_score_F1,i_score_accuracy = compute_interval_scores(gt_seizure_present,pred_seizure_present,gt_onset,pred_onset,max_delay=INTERVAL_DELAY)
    # Berechne Intervall-basierte Metrik (Sommersemester 2023) (höher=besser)
    performance_metric_WS23 = compute_WS23_performance_metric(gt_seizure_present,pred_seizure_present,gt_onset,pred_onset,max_delay=INTERVAL_DELAY)
    
    metrics = {"performance_metric_WS23":performance_metric_WS23,
               "performance_metric_SS23":performance_metric_SS23,
               "detection_error_onset":detection_error_onset,
               "F1":F1,
               "sensitivity":sensitivity,
               "PPV":PPV,
               "accuracy":accuracy,
               "detection_error_offset":detection_error_offset,
               "i_sensitivity":i_score_sensitivity,
               "i_PPV":i_score_PPV,
               "i_accuracy":i_score_accuracy,
               "confusion_matrix":confusion_matrix}
    
    return metrics


def match_predictions(df_gt:pd.DataFrame,df_pred:pd.DataFrame):
    N_files = df_gt.shape[0]
    gt_seizure_present = df_gt[1].to_numpy().astype(bool)
    gt_onset = df_gt[2].to_numpy()
    gt_offset = df_gt[3].to_numpy()
    pred_seizure_present = np.zeros_like(gt_seizure_present,dtype=bool)
    pred_onset = np.zeros_like(gt_onset)
    pred_offset = np.zeros_like(gt_offset)
    pred_seizure_confidence = np.zeros_like(gt_onset)
    pred_onset_confidence = np.zeros_like(gt_onset)
    pred_offset_confidence = np.zeros_like(gt_onset)
    for i in range(N_files):
        _gt_name = df_gt[0][i]
        pred_indx = df_pred[df_pred['id']==_gt_name].index.values

        if not pred_indx.size:
            print("Prediktion für " + _gt_name + " fehlt, nehme \"kein Anfall\" an.")
            pred_seizure_present[i] = False
            pred_seizure_confidence[i] = 0.0
            pred_onset[i] = -1
            pred_onset_confidence[i] = 0.0
            pred_offset[i] = -1
            pred_offset_confidence[i] = 0.0
        else:
            pred_indx = pred_indx[0]
            pred_seizure_present[i] = bool(df_pred['seizure_present'][pred_indx])
            pred_seizure_confidence[i] = df_pred['seizure_confidence'][pred_indx]
            pred_onset[i] = df_pred['onset'][pred_indx]
            if np.isnan(pred_onset[i]):
                pred_onset[i] = 0
            pred_onset_confidence[i] = df_pred['onset_confidence'][pred_indx]
            if np.isnan(pred_onset_confidence[i]):
                pred_onset_confidence[i] = 0
            pred_offset[i] = df_pred['offset'][pred_indx]
            if np.isnan(pred_offset[i]):
                pred_offset[i] = 0
            pred_offset_confidence[i] = df_pred['offset_confidence'][pred_indx]
            if np.isnan(pred_offset_confidence[i]):
                pred_offset_confidence[i] = 0
    return gt_seizure_present,gt_onset,gt_offset,pred_seizure_present,pred_onset,pred_offset,pred_seizure_confidence,pred_onset_confidence,pred_offset_confidence
    
def compute_confusion_matrix(y_gt: np.ndarray,y_pred: np.ndarray) -> List[int]:
    """Berechnet Confusion Matrix basiert auf Ground-Truth und Prediktion

    Args:
        y_gt (np.ndarray): Ground-Truth Werte
        y_pred (np.ndarray): vorhergesagte Werte

    Returns:
        List[4]: [TP,FN,FP,TN]
    """
    assert len(y_gt)==len(y_pred) , "Arrays haben nicht die gleiche Länge"
    TP = np.logical_and(y_gt,y_pred).sum()
    FN = np.logical_and(y_gt,np.logical_not(y_pred)).sum()
    FP = np.logical_and(np.logical_not(y_gt),y_pred).sum()
    TN = np.logical_and(np.logical_not(y_gt),np.logical_not(y_pred)).sum()
    return [int(TP),int(FN),int(FP),int(TN)]


    

def compute_basic_metrics(TP:int,FN:int,FP:int,TN:int) -> Tuple[float,float,float,float]:
    """Berechnet Klassifikationsmetriken Sensitivität, PPV, F1-Score, Accuracy

    Args:
        TP (int): True Positives
        FN (int): False Negatives
        FP (int): False Positives
        TN (int): True Negatives

    Returns:
        Tuple[float,float,float,float]: Sensitivität, PPV, F1, Accuracy
    """
    if (TP+FN) == 0:
        sensitivity = 0
    else:
        sensitivity = TP/(TP+FN)
    if (TP+FP) == 0:
        PPV = 0
    else:
        PPV = TP/(TP+FP) 
    if (sensitivity+PPV) == 0:
        F1 = 0
    else:
        F1 = 2*sensitivity*PPV/(sensitivity+PPV)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    return sensitivity,PPV,F1,accuracy

def compute_detection_error(gt_seizure_present:np.ndarray,pred_seizure_present:np.ndarray,gt_time:np.ndarray,pred_time:np.ndarray,max_penalty=60) -> float: 
    detection_error = (gt_seizure_present&(~pred_seizure_present))*max_penalty \
                    +  (gt_seizure_present&pred_seizure_present)*np.minimum(np.abs(pred_time-gt_time),max_penalty)
    return np.sum(detection_error)/np.sum(gt_seizure_present)

def compute_SS23_performance_metric(detection_error_onset,confusion_matrix,false_classification_pentalty=60):
    N_files = np.sum(confusion_matrix)
    [TP,FN,FP,TN] = confusion_matrix
    N_seizures = TP+FN
    metric = detection_error_onset + (FP/(FP+TN))*false_classification_pentalty*(1-N_seizures/N_files)
    return metric

def compute_WS23_performance_metric(gt_present:np.ndarray,pred_present:np.ndarray,gt_onset:np.ndarray,pred_onset:np.ndarray,max_delay:float=30) -> float:
    """Berechnet die Performance-Metrik des Wintersemester 23/24. 
        Eingangsarrays müssen alle die gleiche Dimension haben.

    Args:
        gt_present (np.ndarray): Ist Anfall vorhanden (GT)
        pred_present (np.ndarray): Ist Anfall vorhanden (Pred)
        gt_onset (np.ndarray): tatsächlicher Onset
        pred_onset (np.ndarray): vorhergesagter Onset
        max_delay (float, optional): Maximaler Abstand zum Tatsächlichen Wert. Defaults to 30.

    Returns:
        float: Performance-Metrik
    """
    return compute_interval_scores(gt_present,pred_present,gt_onset,pred_onset,max_delay)[2]
    
    
def compute_interval_scores(gt_present:np.ndarray,pred_present:np.ndarray,gt_onset:np.ndarray,pred_onset:np.ndarray,max_delay:float=30) -> List[float]:
    """Berechnet Klassifikationsmetriken auf der Basis, ob eine vorhergesagte Zeit in den Bereich der tatsächlichen Zeit fällt
       
       Die Metrik orientiert sich an einer Studie, laut der Patienten und MTAs eine maximale Verzögerung von 30 Sekunden bei der Benachrichtigung über Anfall an
        https://www.sciencedirect.com/science/article/pii/S1059131116301327?pes=vor
    Args:
        gt_present (np.ndarray): Ist Anfall vorhanden (GT)
        pred_present (np.ndarray): Ist Anfall vorhanden (Pred)
        gt_onset (np.ndarray): tatsächlicher Onset
        pred_onset (np.ndarray): vorhergesagter Onset
        max_delay (float, optional): Maximaler Abstand zum Tatsächlichen Wert. Defaults to 30.

    Returns:
        List[float]: [I_Sensitivity, I_PPV, I_F1, I_Accuracy]
    """
    assert gt_present.shape == pred_present.shape == gt_onset.shape == pred_onset.shape , "Arrays haben nicht die gleiche Länge"
    
    in_interval = (np.abs(pred_onset-gt_onset)<max_delay)
    TP = ( (gt_present&pred_present) & in_interval ).sum()
    FN = (((gt_present&pred_present) & (~in_interval)) | (gt_present&(~pred_present))  ).sum()
    FP = ((~gt_present)&pred_present).sum()
    TN = ((~gt_present)&(~pred_present)).sum()
    return compute_basic_metrics(TP,FN,FP,TN)
        

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Predict given Model')
    parser.add_argument('--test_dir', action='store',type=str,default='../test/')
    args = parser.parse_args()
    metrics = score(args.test_dir)
    performance_metric = metrics["performance_metric_WS23"]
    F1 = metrics["F1"]
    detection_error_onset = metrics["detection_error_onset"]
    print("WKI Metrik:", performance_metric,"\t F1:",F1,"\t Latenz:",detection_error_onset)
