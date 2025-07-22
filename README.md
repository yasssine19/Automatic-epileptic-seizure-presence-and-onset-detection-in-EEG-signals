# Seizure Detection using EEG and Machine Learning
**"AI in medicine" - Summer Semester 2025**

# Competitive Overview
This Repository was developed as part of the **"KI in der Medizin" (AI in Medicine)** seminar at TU Darmstadt in the summer semester of 2025. A total of 6 teams participated, each consisting of 3 members. Our Team achieved the **1st place** by building the most accurate seizure detection system evaluated on a hidden test set.

---

## Dataset: Temple University Hospital EEG Corpus (TUH)

- Total: **6213 EEG samples**
- Origin: **Temple University Hospital**
- Not publicly shareable via GitHub due to licensing restrictions.
- Each sample contains EEG sequences from a patient (patients may appear in multiple samples).
- Most samples have **19 EEG channels**, a few (~200) contain only **17 channels**.
- We compute **20 bipolar EEG montages** (e.g., `Fp1-F3`) from the available channels, which are used for feature extraction and classification.

---

## Data Splitting

- **Training Set**: 80%
- **Test Set**: 20%
- For training, we only use **seizure-containing samples (class 1)** to assure class balance in sequences.
- Resulting in:
  - **Training**: 405.553 windows (30% class 1, 70% class 0)
  - **Testing**: 392.105 windows (contains both class 0 and class 1)

---

## Feature Extraction

Each EEG signal is segmented into **4-second windows** with **1-second overlap**. From each window and for each of the **20 montages**, we extract the following **22 features**, resulting in a **20×22 feature vector** per sample:

### Time Domain Features
- Mean, Variance, Skewness, Kurtosis, IQR, Minimum, Maximum
- Monotonicity (zero-crossings in signal)
- **Hjorth Parameters**: Mobility, Complexity
- **Petrosian Fractal Dimension**

### Frequency Domain Features
Computed using Welch's method (with sampling frequency `fs`):
- **Band Powers**:
  - Delta (1–4 Hz)
  - Theta (4–8 Hz)
  - Alpha (8–13 Hz)
  - Beta  (13–35 Hz)
  - Gamma (35–70 Hz)
- **Power Ratios** for each band
- **Spectral Centroid**

All features are extracted **in parallel** and saved as `.npy` files to speed up processing.

---

## Machine Learning Model

We train a **HistGradientBoostingClassifier** using the extracted features.

- **Hyperparameter tuning** via `RandomizedSearchCV`
- The final model is exported and reused for inference on the test set.
- Predictions are made **per window**, across **all montages** of each sample.

---

## Postprocessing

To improve robustness, we apply two postprocessing steps:

1. **Noise Filtering**: Predicted seizure windows that are not part of a sequence of **at least 7 consecutive windows** labeled as seizure (1) are removed.
2. **Onset Detection**: From the longest sequence of seizure predictions, the **first window** is selected as the **predicted onset time**.

Final output per sample:
- **Seizure Presence**: `0` (absent) or `1` (present)
- **Onset Time**:
  - If no seizure: `0.0`
  - If seizure detected: timestamp of first window in longest 1-sequence

---

## Evaluation Strategy

We evaluate the model in three ways:

### 1. **Window-wise Evaluation**
- Metrics: Precision, Recall, F1-score
- Based only on presence of seizures (onset not considered)

### 2. **Sequence-wise Evaluation**
- Aggregated evaluation on a per-sequence level (binary classification of seizure presence)

### 3. **Onset-aware Evaluation**
- Considers both seizure presence and correctness of the predicted **onset time**:
  - **TP**: Seizure correctly detected, and onset is within ±30 seconds of ground truth.
  - **FP**: Seizure predicted when none exists.
  - **FN**: Seizure missed or onset outside allowed margin.
  - **TN**: Correctly predicted as no seizure.

We also report:
- **Latency**: `min(abs(predicted_onset - true_onset), 60)`  
  _(capped at 60s to avoid over-penalization)_

Note: The **±30s margin** is based on medical standards for acceptable onset detection tolerance.

---

## Highlights

- Parallelized feature extraction for high performance.
- Use of handcrafted statistical and spectral features tailored for EEG data.
- Model deployment with `.npy` I/O pipelines and robust postprocessing.
- Best performance on hidden evaluation set among all participating teams.

---

## Getting started

To use this repository and test our results, follow these steps:

1. **Clone or download** this repository to your local machine.

2. **Create a Python environment** (for example [Anaconda](https://www.anaconda.com/products/distribution)):
```
 conda activate eeg-processing
 pip install -r requirements.txt
``` 
3. **Activate the environment** and install all required packages:
```
conda activate eeg-processing
pip install -r requirements.txt
```


## Licensing & Ethical Note

- The TUH dataset is under **restricted license** and must **not** be uploaded to public repositories.
- All experiments were conducted in compliance with the dataset's terms of use.

---
