# Tutorial: Preprocessing Stanford FingerFlex Dataset

This document provides a detailed tutorial for preprocessing the Stanford FingerFlex dataset and preparing it for a neural decoding prediction task.

## 1. Dataset Acquisition
The Stanford FingerFlex dataset consists of ECoG recordings from 9 subjects while they performed finger flexion tasks.
* **Download Link:** [Stanford-FingerFlex: zk881ps0522](https://searchworks.stanford.edu/view/zk881ps0522)
* **Storage:** Download the raw subject folders and place them under a directory of your choice (e.g. `$CORTEG_DATA_ROOT/raw/Stanford/`). Update the `dataLoc` variable at the top of `data_preprocessing_Stanford.m` to point to that path.

## 2. Environment Setup
The pipeline requires two distinct environments:

### MATLAB Environment
* **Software:** MATLAB R2022b.
* **Toolbox:** FieldTrip (version fieldtrip-20230926).

### Python Environment
* **Toolbox:** MNE (v1.11.0).
* **Libraries:** `scipy`, `numpy`, `pickle`.

---

## 3. Step 1: MATLAB Signal Preprocessing
Run the script `data_preprocessing_Stanford.m` to clean the raw ECoG signals.

### Key Operations:
* **Scaling:** Data is scaled by a factor of 0.0298 to adjust units. 
* **Line Noise Removal:** Bandstop filters are applied at 60Hz, 120Hz, and 180Hz (bandwidth of 1Hz each) using a Butterworth filter.
* **Interactive Bad Channel Selection:** The script uses `ft_databrowser` to visualize the data. You must manually identify faulty channels (e.g., 'ch40' for subject 'mv') and enter them in the prompt.
* **Common Average Referencing (CAR):** All remaining channels are re-referenced to the global average to reduce common noise.
* **Output:** Preprocessed `.mat` files are saved to `./preprocessed_data/Stanford/`.

---

## 4. Step 2: Python Task Formatting & Feature Extraction
Run `prepare_taskFormatedData_Stanford.py` to prepare the data for machine learning models.

### Feature Engineering (`HGALFS_feature_extractor`):
For each epoch, the following features are extracted from 1-second ECoG windows:
1.  **High Gamma Activity (HGA):**
    * Filtered between 70Hz and 200Hz.
    * Envelope extracted via Hilbert transform.
    * Downsampled to 200 time points ($T=200$).
2.  **Low Frequency Signal (LFS):**
    * Raw ECoG signal downsampled to 200Hz. 
3.  **Temporal Alignment:**
    * A **40ms delay** is applied between the neural window and the target trajectory to account for delay.

### Dataset Splitting:
The script splits the data chronologically:
* **Long recordings (>= 600s):** First 400s for training, the rest for testing.
* **Short recordings:** 2/3 for training, 1/3 for testing.

---

## 5. Final Output Data Structure
The processing generates a `.pkl` file for each subject containing a list with the following components:

| Variable | Description | Dimensions |
| :--- | :--- | :--- |
| `ECoG_train` | Combined HGA and LFS features (Train) | `[nEpoch, nChannel, 200, 2]` |
| `trajectory_train` | Finger flexion ground truth (Train) | `[nEpoch, 5]` |
| `ECoG_test` | Combined HGA and LFS features (Test) | `[nEpoch, nChannel, 200, 2]` |
| `trajectory_test` | Finger flexion ground truth (Test) | `[nEpoch, 5]` |
| `ECoG_train_128` | ECoG downsampled to 128Hz (Train) | `[nEpoch, nChannel, 128]` |
| `ECoG_test_128` | ECoG downsampled to 128Hz (Test) | `[nEpoch, nChannel, 128]` |

---
**References:**
1. Miller, Kai J. "A library of human electrocorticographic data and analyses." Nature human behaviour 3.11 (2019): 1225-1235.
2. Miller, Kai J., et al. "Human motor cortical activity is selectively phase-entrained on underlying rhythms." (2012): e1002655.
