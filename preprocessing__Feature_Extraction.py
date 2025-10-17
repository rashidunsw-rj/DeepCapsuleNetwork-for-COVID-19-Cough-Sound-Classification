# Placeholder for original script content
# Assume the rest of the feature extraction and augmentation code is already here
import librosa
import time
import os
import numpy as np
from datetime import datetime
import pandas as pd
from scipy.stats import kurtosis, skew


path = 'C:\MSC\COUVI'


lst = []
start_time = time.time()

def pitch(data, sample_rate):
    """
    Apply pitch tuning to the audio signal.
    """
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform() - 0.5)  # Random shift within range

    # Ensure the input is of type float64
    data = data.astype(np.float64)

    # Apply pitch shift
    data = librosa.effects.pitch_shift(data, sr=sample_rate, 
                                       n_steps=pitch_change, 
                                       bins_per_octave=bins_per_octave)
    return data


def noise(data):

    noise_amp = 0.005*np.random.uniform()*np.amax(data)
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data



def stretch(data, rate=0.8):
    """
    Stretch the audio without changing pitch.
    
    Parameters:
    - data (np.array): 1D NumPy array of audio data
    - rate (float): Speed factor (e.g., <1.0 slows down, >1.0 speeds up)
    
    Returns:
    - np.array: Time-stretched audio
    """
    # Ensure data is a float64 NumPy array
    data = np.ascontiguousarray(data, dtype=np.float64)

    # Ensure data is mono (1D)
    if data.ndim > 1:
        data = librosa.to_mono(data)

    # Apply time stretch
    return librosa.effects.time_stretch(y=data, rate=rate)





"""
Read each file in dataset and extract the features
"""
for subdir, dirs, files in os.walk(path):

   for file in files:
      try:
        em = int(os.path.basename(subdir))  # Use folder name as class label (0, 1, 2, 3)
        file_path = os.path.join(subdir, file)
        print(f"Processing file: {file_path}, Label: {em}")

        X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')       
        Y=pitch(X,sample_rate)
        Z=noise(X)
        A=stretch(X,0.8)
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=60)
        mfccs_Y = librosa.feature.mfcc(y=Y, sr=sample_rate, n_mfcc=60)
        mfccs_Z = librosa.feature.mfcc(y=Z, sr=sample_rate, n_mfcc=60)
        mfccs_A = librosa.feature.mfcc(y=A, sr=sample_rate, n_mfcc=60)
        
        
        kurt = kurtosis(X)
        skewness = skew(X)
        
        kurt_Y = kurtosis(Y)
        skewness_Y = skew(Y)
        
        kurt_Z = kurtosis(Z)
        skewness_Z = skew(Z)
        
        kurt_A = kurtosis(A)
        skewness_A = skew(A)
        


        rms = np.mean(librosa.feature.rms(y=X).T,axis=0)
        rms_Y = np.mean(librosa.feature.rms(y=X).T,axis=0)
        rms_Z = np.mean(librosa.feature.rms(y=X).T,axis=0)
        rms_A = np.mean(librosa.feature.rms(y=X).T,axis=0)
        
        
        mfccs =np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=60).T, axis=0)
        mfccs_Y =np.mean(librosa.feature.mfcc(y=Y, sr=sample_rate, n_mfcc=60).T, axis=0)
        mfccs_Z =np.mean(librosa.feature.mfcc(y=Z, sr=sample_rate, n_mfcc=60).T, axis=0)
        mfccs_A =np.mean(librosa.feature.mfcc(y=A, sr=sample_rate, n_mfcc=60).T, axis=0)
        
 
        
        stft=np.abs(librosa.stft(X))
        stft_Y=np.abs(librosa.stft(Y))
        stft_Z=np.abs(librosa.stft(Z))
        stft_A=np.abs(librosa.stft(A))
      
        
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        tonnetz=np.mean(librosa.feature.tonnetz(y=X,sr=sample_rate).T,axis=0)
        contrast=np.mean(librosa.feature.spectral_contrast(y=X,sr=sample_rate).T,axis=0)
        
        chroma_Y = np.mean(librosa.feature.chroma_stft(S=stft_Y, sr=sample_rate).T,axis=0)
        mel_Y = np.mean(librosa.feature.melspectrogram(y=Y, sr=sample_rate).T,axis=0)
        tonnetz_Y=np.mean(librosa.feature.tonnetz(y=Y,sr=sample_rate).T,axis=0)
        contrast_Y=np.mean(librosa.feature.spectral_contrast(y=Y,sr=sample_rate).T,axis=0)
        
        chroma_Z = np.mean(librosa.feature.chroma_stft(S=stft_Z, sr=sample_rate).T,axis=0)
        mel_Z = np.mean(librosa.feature.melspectrogram(y=Z, sr=sample_rate).T,axis=0)
        tonnetz_Z=np.mean(librosa.feature.tonnetz(y=Z,sr=sample_rate).T,axis=0)
        contrast_Z=np.mean(librosa.feature.spectral_contrast(y=Z,sr=sample_rate).T,axis=0)
        
        chroma_A = np.mean(librosa.feature.chroma_stft(S=stft_A, sr=sample_rate).T,axis=0)
        mel_A = np.mean(librosa.feature.melspectrogram(y=A, sr=sample_rate).T,axis=0)
        tonnetz_A=np.mean(librosa.feature.tonnetz(y=A,sr=sample_rate).T,axis=0)
        contrast_A=np.mean(librosa.feature.spectral_contrast(y=A,sr=sample_rate).T,axis=0)
        
       
        
    
        features=np.hstack((mfccs,chroma,mel,tonnetz,contrast,mfcc_delta,mfcc_delta2,kurt,skewness,rms)).T
        arr = features, em
        features_Y=np.hstack((mfccs_Y,chroma_Y,mel_Y,tonnetz_Y,contrast_Y,kurt_Y,skewness_Y,rms_A)).T
        arr_Y = features_Y, em
        features_Z=np.hstack((mfccs_Z,chroma_Z,mel_Z,tonnetz_Z,contrast_Z,kurt_Z,skewness_Z,rms_Z)).T
        arr_Z = features_Z, em
        features_A=np.hstack((mfccs_A,chroma_A,mel_A,tonnetz_A,contrast_A,kurt_A,skewness_A,rms_A)).T
        arr_A = features_A, em
       
        
        lst.append(arr)
        lst.append(arr_Y)
        lst.append(arr_Z)
        lst.append(arr_A)
      # If the file is not valid, skip it
      except ValueError:
        continue
X, y = zip(*lst)
X = np.asarray(X)
y = np.asarray(y)

X.shape, y.shape

import joblib

X_name = 'COUGH4X.joblib'
y_name = 'COUGH4Y.joblib'
save_dir = 'C:\MSC'

savedX = joblib.dump(X, os.path.join(save_dir, X_name))
savedy = joblib.dump(y, os.path.join(save_dir, y_name))

# ====================== SMOTE + NaN Handling ======================
from imblearn.over_sampling import SMOTE


valid = ~np.isnan(X).any(axis=1)
X = X[valid]
y = y[valid]


sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)


# Optional: Save balanced output
joblib.dump(X, os.path.join(save_dir, 'COUGH4X_SMOTE.joblib'))
joblib.dump(y, os.path.join(save_dir, 'COUGH4Y_SMOTE.joblib'))
