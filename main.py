# -*- coding: utf-8 -*-

import os
import glob
from predict import run_test_predictions
from classifier_model import ClassifierModel
from preprocessing import process_records_parallel, balanced_patient_split, process_reference_data, load_data

OUT_DIR="../output"
TRAIN_DIR = os.path.join(OUT_DIR, "train")
TEST_DIR = os.path.join(OUT_DIR, "test")
data_folder = "../shared_data/training"

if __name__ == '__main__':
    """
    Führt die End-to-End-Pipeline aus: Daten vorbereiten, trainieren, evaluieren, speichern.
    """
    # Nur Patienten mit mindestens einem Sample jeder Klasse beibehalten
    ref_df, eligible_patients = process_reference_data(data_folder)

    train_records, test_records = balanced_patient_split(ref_df, eligible_patients)

    process_records_parallel(train_records, TRAIN_DIR)
    process_records_parallel(test_records, TEST_DIR)

    # Training-Datensatz laden
    train_feature_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*_feats.npy")))
    train_label_files   = sorted(glob.glob(os.path.join(TRAIN_DIR, "*_labels.npy")))
    assert len(train_feature_files) == len(train_label_files), "Mismatch in number of feature and label files."
    X_train,y_train= load_data(train_feature_files,train_label_files)

    # Test-Datensatz laden
    test_feature_files = sorted(glob.glob(os.path.join(TEST_DIR, "*_feats.npy")))
    test_label_files   = sorted(glob.glob(os.path.join(TEST_DIR, "*_labels.npy")))
    assert len(test_feature_files) == len(test_label_files), "Mismatch in number of feature and label files."
    X_test, y_test = load_data(test_feature_files, test_label_files)

    
    classifier_model = ClassifierModel()
    classifier_model.train(X_train, y_train)
    classifier_model.evaluate(X_train, y_train, label="Training")
    test_metrics = classifier_model.evaluate(X_test, y_test, label="Test")
    test_results = run_test_predictions(test_records, classifier_model, TEST_DIR, min_consecutive=8)
    classifier_model.compute_interval_metrics(test_results)    
    classifier_model.save("choose_your_name.pkl")