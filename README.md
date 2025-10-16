# DeepCapsNet for COVID-19 Cough Sound Classification

## Description
This package provides the **trained DeepCapsNet models** developed for classifying cough sounds as **Healthy**, **COVID-19**, or **Symptomatic** using acoustic features extracted from the **COSWARA** and **COUGHVID** datasets.  
Only trained models and evaluation-related materials are shared to support reproducibility and verification while preserving proprietary training implementation.

## Dataset Information
The following publicly available datasets were used for training and evaluation:
- **COSWARA Dataset**:  https://www.kaggle.com/datasets/janashreeananthan/coswara
- **COUGHVID Dataset**: https://www.kaggle.com/datasets/nasrulhakim86/coughvid-wav

Each cough audio was preprocessed, augmented (pitch shift, noise, stretching), and transformed into Mel-frequency cepstral coefficients (MFCCs) and related acoustic features before training.

## Contents
| File/Folder | Description |
|--------------|-------------|
| `saved_models_coughvidaug/` | Directory containing pre-trained DeepCapsNet `.h5` models (one per fold). |
| `Materials_and_Methods.txt` | Detailed explanation of datasets, preprocessing, and evaluation. |
| `requirements.txt` | Python environment dependencies. |

## Usage Instructions
### 1. Load the Trained Models
```python
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load model
model = load_model("saved_models_coughvidaug/coughvidaugdcap_model_fold_1.h5")

# Load data (if provided)
X = joblib.load("COVID7X_SMOTE.joblib")
y = joblib.load("COVID7Y_SMOTE.joblib")

# Predict
predictions = np.argmax(model.predict(X), axis=1)
```

### 2. Evaluate Model
You can use `classification_report_coughvidaug.txt` or regenerate metrics using scikit-learn if evaluation data are available.

## Requirements
| Package | Version |
|----------|----------|
| Python | 3.8+ |
| TensorFlow | 2.8+ |
| NumPy | 1.21+ |
| scikit-learn | 1.0+ |
| Matplotlib | 3.5+ |
| Joblib | 1.1+ |
| SciPy | 1.7+ |

Install dependencies:
```bash
pip install -r requirements.txt
```

## Methodology Summary
1. **Feature Extraction:** MFCCs, chroma, mel-spectrogram, tonnetz, and spectral contrast.  
2. **Data Balancing:** SMOTE for class balance.  
3. **Model Architecture:** DeepCapsNet combining 1D CNN and Capsule layers.  
4. **Training:** Stratified K-Fold cross-validation.  
5. **Evaluation:** Accuracy, confusion matrix, RMSE, G-Mean, and 95% confidence intervals.

## Citations
- Sharma et al., “Coswara: A Database of Breathing, Cough, and Voice Sounds for COVID-19 Diagnosis,” *arXiv:2005.10548*.  
- Orlandic et al., “COUGHVID: A Crowdsourced Dataset for COVID-19 Cough Classification,” *arXiv:2006.04342*.

## License
The trained model files are shared under an **academic non-commercial license** for verification and research purposes only.
