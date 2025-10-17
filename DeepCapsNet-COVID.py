# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:46:32 2025

@author: rashi
"""



import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.layers import Input, Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# Load Data
# ---------------------------
X = joblib.load('F:/PhD/COVID7X_SMOTE.joblib')
y = joblib.load('F:/PhD/COVID7Y_SMOTE.joblib')
from sklearn.preprocessing import LabelEncoder

# Convert y to a numpy array if not already
y = np.array(y)

# Encode string labels to integers
le = LabelEncoder()
y = le.fit_transform(y)

# Expand dims for Conv1D
X = np.expand_dims(X, axis=2)  # shape: (samples, features, 1)

# ---------------------------
# Cross-validation setup
# ---------------------------
n_splits = 2
batch_size = 8
epochs = 1

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold = 10
all_fold_accuracies = []
history_list = []
conf_matrices = []

for train_index, val_index in skf.split(X, y):
    print(f"===== Fold {fold} =====")

    X_train_fold = X[train_index]
    y_train_fold = y[train_index]
    X_val_fold = X[val_index]
    y_val_fold = y[val_index]

    # ---------------------------
    # Build CNN Model
    # ---------------------------
    input_layer = Input(shape=(X.shape[1], 1))

    x = Conv1D(64, 5, padding='same', activation='relu')(input_layer)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(pool_size=4)(x)

    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(pool_size=4)(x)

    x = Conv1D(256, 5, padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(pool_size=4)(x)

    x = Conv1D(512, 3, padding='same', activation='relu', name='last_conv1d')(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    
    # Dense Capsule Layer
    dense_caps = layers.Dense(16)(x)  # 16-dimensional capsules
    dense_caps = layers.Lambda(squash)(dense_caps)
    
    # Lambda Capsule Layer (L2 normalization)
    lambda_caps = layers.Lambda(lambda x: tf.norm(x, axis=-1))(dense_caps)
    
    # Output Layer
    output = layers.Dense(len(np.unique(y)), activation='softmax')(lambda_caps)
 

    model = Model(inputs=input_layer, outputs=output)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    # ---------------------------
    # Training
    # ---------------------------
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train_fold, y_train_fold,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val_fold, y_val_fold),
        callbacks=[early_stop],
        verbose=2
    )
    history_list.append(history.history)

    # ---------------------------
    # Evaluation
    # ---------------------------
    scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    print(f"Fold {fold} accuracy: {scores[1]:.4f}")
    all_fold_accuracies.append(scores[1])

    # Confusion matrix
    y_pred = np.argmax(model.predict(X_val_fold), axis=1)
    cm = confusion_matrix(y_val_fold, y_pred)
    conf_matrices.append(cm)

    fold += 1

# ---------------------------
# Mean Accuracy
# ---------------------------
print(f"\nMean accuracy over {n_splits} folds: {np.mean(all_fold_accuracies):.4f}")
print(f"Std deviation: {np.std(all_fold_accuracies):.4f}")

# ---------------------------
# Plot Training/Validation Loss and Accuracy (Average)
# ---------------------------

avg_loss = np.mean([h['loss'] for h in history_list], axis=0)
avg_val_loss = np.mean([h['val_loss'] for h in history_list], axis=0)
avg_acc = np.mean([h['accuracy'] for h in history_list], axis=0)
avg_val_acc = np.mean([h['val_accuracy'] for h in history_list], axis=0)





plt.figure(figsize=(12, 8))
for i, history in enumerate(history_list, 1):
    plt.plot(history['loss'], label=f'Fold {i} Train Loss')
    plt.plot(history['val_loss'], linestyle='--', label=f'Fold {i} Val Loss')

plt.ylabel('loss (x 100%)')
plt.xlabel('Number of epochs')
#plt.title('Training and Validation Loss per Fold')
plt.legend()
plt.gca().set_facecolor('#e0e0e0')
plt.grid(color='white') 
plt.savefig('coughvidaug_each_fold_loss.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()


# ---------------------------
# Plot Accuracy for Each Fold
# ---------------------------
plt.figure(figsize=(12, 8))
for i, history in enumerate(history_list, 1):
    plt.plot(history['accuracy'], label=f'Fold {i} Train Acc')
    plt.plot(history['val_accuracy'], linestyle='--', label=f'Fold {i} Val Acc')

plt.ylabel('loss (x 100%)')
plt.xlabel('Number of epochs')
#plt.title('Training and Validation Loss per Fold')
plt.legend()
plt.gca().set_facecolor('#e0e0e0')
plt.grid(color='white') 
plt.savefig('coughvidaug_each_fold_accuracy.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()



# ---------------------------
# Plot Mean Confusion Matrix
# ---------------------------
mean_cm = np.mean(conf_matrices, axis=0)
fig2, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=mean_cm)
disp.plot(ax=ax, cmap='Blues', values_format=".2f")
plt.title('Mean Confusion Matrix (10-fold)')
plt.savefig('coughvid_confusion_matrix.png', dpi=600)
plt.show()

print(mean_cm)


mean_cm = np.mean(conf_matrices, axis=0)
mean_cm_percent = (mean_cm / mean_cm.sum(axis=1, keepdims=True)) * 100
print(mean_cm)

# Define labels
labels = ['Healthy', 'COVID', 'Symptomatic']

# Plot
fig2, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=mean_cm_percent, display_labels=labels)
disp.plot(ax=ax, cmap='Blues', values_format=".1f")  # Show one decimal percentage
plt.title('Mean Confusion Matrix (10-fold) [%]')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.savefig('coughvidaug_confusion_matrix_percentage.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()



from sklearn.metrics import classification_report


# Store predictions and true labels for all folds
all_y_true = []
all_y_pred = []

fold = 1
for train_index, val_index in skf.split(X, y):
    print(f"===== Fold {fold} =====")
    
    # ... your training code ...
    
    # After predictions
    y_pred = np.argmax(model.predict(X_val_fold), axis=1)
    
    all_y_true.extend(y_val_fold)
    all_y_pred.extend(y_pred)
    
    fold += 1

# Convert to numpy arrays
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)


labels = ['Healthy', 'COVID', 'Symptomatic']

# Generate classification report
report = classification_report(all_y_true, all_y_pred, target_names=labels, digits=4)
print(report)

# Save to file
with open("classification_report_coughvidaug.txt", "w") as f:
    f.write(report)
    
    
    



import numpy as np
from sklearn.metrics import confusion_matrix
from math import sqrt
labels = ['Healthy', 'COVID', 'Symptomatic']
n_classes = len(labels)


rmse_per_class = []
for i in range(n_classes):
    y_true_binary = (all_y_true == i).astype(int)
    y_pred_binary = (all_y_pred == i).astype(int)

    mse = np.mean((y_true_binary - y_pred_binary) ** 2)
    rmse = sqrt(mse)
    rmse_per_class.append(rmse)

# -------------------------------
# 2️⃣ Compute G-Mean for each class
# -------------------------------
cm = confusion_matrix(all_y_true, all_y_pred, labels=np.arange(n_classes))
gmean_per_class = []

for i in range(n_classes):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - (TP + FN + FP)

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    gmean = sqrt(sensitivity * specificity)
    gmean_per_class.append(gmean)

# -------------------------------
# Print results
# -------------------------------
print(" Class-wise RMSE:")
for label, rmse in zip(labels, rmse_per_class):
    print(f"{label}: {rmse:.4f}")

print("\n Class-wise G-Mean:")
for label, gmean in zip(labels, gmean_per_class):
    print(f"{label}: {gmean:.4f}")




import numpy as np
import scipy.stats as st
from sklearn.metrics import confusion_matrix


labels = ['Healthy', 'COVID', 'Symptomatic']
n_classes = len(labels)


per_class_acc = np.zeros((len(conf_matrices), n_classes))

for i, cm in enumerate(conf_matrices):
    cm = np.array(cm)
    class_acc = np.diag(cm) / cm.sum(axis=1)  # accuracy = TP / (TP+FN)
    per_class_acc[i] = class_acc

# Compute statistics
mean_acc_per_class = np.mean(per_class_acc, axis=0)
std_dev_per_class = np.std(per_class_acc, axis=0, ddof=1)

confidence_level = 0.95
degrees_freedom = len(conf_matrices) - 1
ci_per_class = np.array([
    st.t.interval(confidence_level, degrees_freedom, 
                  loc=mean_acc_per_class[i], 
                  scale=st.sem(per_class_acc[:, i]))
    for i in range(n_classes)
])

# Print results
for i, label in enumerate(labels):
    print(f"{label}: Mean Accuracy = {mean_acc_per_class[i]:.4f}, "
          f"Std Dev = {std_dev_per_class[i]:.4f}, "
          f"95% CI = ({ci_per_class[i,0]:.4f}, {ci_per_class[i,1]:.4f})")



import os
import joblib

save_dir = "saved_models_coughvidaug"
os.makedirs(save_dir, exist_ok=True)

fold = 1
for train_index, val_index in skf.split(X, y):
    print(f"===== Fold {fold} =====")
  

    # Save model
    model_save_path = os.path.join(save_dir, f"coughvidaugdcap_model_fold_{fold}.h5")
    model.save(model_save_path)
    print(f"Saved model for Fold {fold} at {model_save_path}")


    fold += 1


results = {
    "all_fold_accuracies": all_fold_accuracies,
    "conf_matrices": conf_matrices,
    "history_list": history_list
}
joblib.dump(results, os.path.join(save_dir, "coughvidaug_crossval_results.joblib"))
print("All models and training results saved successfully!")


