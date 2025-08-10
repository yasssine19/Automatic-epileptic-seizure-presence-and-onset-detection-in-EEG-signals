# -*- coding: utf-8 -*-

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

class ClassifierModel:

    """
    Wrapper-Klasse für einen `HistGradientBoostingClassifier` mit
    bequemen Hilfsmethoden für Training, Evaluation, Persistenz und
    onset-bewusste Intervallmetriken.

    Parameter
    ---------
    model : Optional[HistGradientBoostingClassifier], default=None
        Ein bereits instanziiertes (ggf. trainiertes) Sklearn-Modell.
        Wenn `None`, wird intern ein `HistGradientBoostingClassifier`
        mit (getunten) Default-Hyperparametern erstellt.
    model_params : Optional[dict], default=None
        Überschreibt die Standard-Hyperparameter, falls gesetzt.
        Wird nur verwendet, wenn `model is None`.
    model_name : str, default="Ace"
        Name des Modells, u. a. für den Dateinamen beim Speichern/Laden.

    Hinweise
    --------
    - Die gewählten Default-Hyperparameter spiegeln euer Tuning wider.
    - `class_weight="balanced"` kann bei Klassendiskrepanzen helfen.
    """

    def __init__(self, model=None, model_params=None, model_name="Ace"):
        """
        Initialisiert das Modellobjekt.

        Wenn kein externes Modell übergeben wurde, wird ein
        `HistGradientBoostingClassifier` mit sinnvollen Defaults erzeugt.
        """
        if model is not None:
            self.model = model
        else:
            # Unsere gewählten Hyperparameter nach dem Tuning
            params = model_params if model_params is not None else {
                "max_iter": 250,
                "learning_rate": 0.025,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 50,
                "l2_regularization": 0.1,
                "class_weight": "balanced"
            }
            self.model = HistGradientBoostingClassifier(**params)
        self.model_name = model_name

    def train(self, X_train, y_train):
        """
        Trainiert den Klassifikator auf den übergebenen Trainingsdaten.

        Parameter
        ---------
        X_train : array-like, shape (n_samples, n_features)
            Feature-Matrix des Trainingssatzes.
        y_train : array-like, shape (n_samples,)
            Ziel-Labels (0/1) für den Trainingssatz.

        Rückgabewert
        ------------
        None
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X, y, label=""):
        """
        Evaluiert das Modell auf beliebigen Daten und gibt Metriken zurück.

        Es werden Accuracy sowie Precision/Recall/F1 pro Klasse berechnet.
        Zusätzlich wird der vollständige `classification_report` geliefert.

        Parameter
        ---------
        X : array-like, shape (n_samples, n_features)
            Feature-Matrix für die Evaluation.
        y : array-like, shape (n_samples,)
            Ground-Truth-Labels (0/1) für die Evaluation.
        label : str, optional
            Freitext, der in der Ausgabe vorangestellt wird (z. B. "Test").

        Rückgabewert
        ------------
        dict
            Dictionary mit folgenden Keys:
            - "accuracy" : float
            - "precision" : List[float]   (pro Klasse [0, 1])
            - "recall" : List[float]      (pro Klasse [0, 1])
            - "f1_score" : List[float]    (pro Klasse [0, 1])
            - "report" : dict             (sklearn classification_report als dict)
        """
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"{label} Accuracy: {acc:.4f}")
        report = classification_report(y, y_pred, output_dict=True)
        precision = [report['0']['precision'], report['1']['precision']]
        recall = [report['0']['recall'], report['1']['recall']]
        f1_score_ = [report['0']['f1-score'], report['1']['f1-score']]
        print(" Precision:", precision)
        print(" Recall:", recall)
        print(" F1-Score:", f1_score_)
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score_,
            "report": report
        }
    
    def predict(self, X):
        """
        Gibt Klassenlabels für die übergebenen Beispiele zurück.

        Parameter
        ---------
        X : array-like, shape (n_samples, n_features)
            Feature-Matrix für die Vorhersage.

        Rückgabewert
        ------------
        np.ndarray, shape (n_samples,)
            Vorhergesagte Klassenlabels (0/1).
        """
        return self.model.predict(X)
    
    def save(self, path=None):
        """
        Speichert das interne Sklearn-Modell via `joblib.dump`.

        Parameter
        ---------
        path : Optional[str], default=None
            Zielpfad der Modell-Datei. Standard: `<model_name>.pkl`.

        Rückgabewert
        ------------
        None
        """
        if path is None:
            path = f"{self.model_name}.pkl"
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, path=None):
        """
        Lädt ein zuvor gespeichertes Sklearn-Modell via `joblib.load`.

        Parameter
        ---------
        path : Optional[str], default=None
            Pfad zur Modell-Datei. Standard: `<model_name>.pkl`.

        Rückgabewert
        ------------
        None
        """
        if path is None:
            path = f"{self.model_name}.pkl"
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")

    def compute_interval_metrics(self, test_results, max_delay=30.0):
        """
        Berechnet onset-bewusste Intervallmetriken inkl. Latenz.

        Diese Metrik bewertet *sowohl* die Präsenz einer Seizure als auch,
        ob der vorhergesagte Onset innerhalb eines klinisch sinnvollen Fensters
        (± `max_delay` Sekunden) um den Referenz-Onset liegt.

        Zählt Fälle wie folgt:
        - **TP** (True Positive): `true_label=1` und `pred_label=1`,
          *und* |pred_onset - true_onset| ≤ max_delay.
        - **FP** (False Positive): `true_label=0`, `pred_label=1`.
        - **FN** (False Negative):
            a) `true_label=1`, `pred_label=0` (übersehen),
            b) `true_label=1`, `pred_label=1`, aber Onset > max_delay entfernt.
        - **TN** (True Negative): `true_label=0`, `pred_label=0`.

        Zusätzlich werden zwei Latenz-Statistiken berechnet:
        - **latency_true**: mittlere |pred - true|, gedeckelt bei 60 s
          (vermeidet übermäßige Bestrafung sehr großer Fehler).
        - **latency_without_min**: mittlere |pred - true| ohne Deckelung.

        Parameter
        ---------
        test_results : Iterable[Tuple[record_id, pred_label, pred_onset, true_label, true_onset]]
            Sequenz von Ergebnissen pro Aufzeichnung:
            - record_id : beliebiger Identifier (int/str)
            - pred_label : int (0/1) - vorhergesagte Präsenz
            - pred_onset : float - vorhergesagter Onset (Sekunden)
            - true_label : int (0/1) - Ground-Truth-Präsenz
            - true_onset : float - Referenz-Onset (Sekunden)
        max_delay : float, default=30.0
            Toleranzfenster (Sekunden) für eine „zeitlich korrekte“ Detektion.

        Rückgabewert
        ------------
        None
        """
        I_TP = I_FP = I_FN = I_TN = 0
        latency_true = latency_without_min = 0
        for (record_id, pred_label, pred_onset, true_label, true_onset) in test_results:
            if true_label == 1 and pred_label == 1:
                latency_true += min(abs(pred_onset - true_onset), 60)
                latency_without_min += abs(pred_onset - true_onset)
                # True seizure, predicted seizure
                if abs(pred_onset - true_onset) <= max_delay:
                    I_TP += 1  # correct and timely detection
                else:
                    I_FN += 1  # predicted, but onset far off (count as miss)
            elif true_label == 1 and pred_label == 0:
                I_FN += 1      # missed a seizure
            elif true_label == 0 and pred_label == 1:
                I_FP += 1      # false alarm (no true seizure)
            elif true_label == 0 and pred_label == 0:
                I_TN += 1      # correct rejection

        # Interval-based Precision, Recall, F1
        I_precision = I_TP / (I_TP + I_FP) if (I_TP + I_FP) > 0 else 0.0
        I_recall = I_TP / (I_TP + I_FN) if (I_TP + I_FN) > 0 else 0.0
        I_f1 = 2*I_TP / (2*I_TP + I_FP + I_FN) if I_TP > 0 else 0.0

        print('true positive: ', I_TP)
        print('false positive: ', I_FP)
        print('false negative: ', I_FN)
        print('true negative: ', I_TN)
        print()
        print('latency_without_min = ', latency_without_min/len(test_results))
        print()
        print('latency_true = ', latency_true/len(test_results))
        print(f"Interval-based Precision: {I_precision:.3f}")
        print(f"Interval-based Recall: {I_recall:.3f}")
        print(f"Interval-based F1-score (WS23 Performance Metric): {I_f1:.3f}")