"""
================================================================================
  GENAI SKILL LAB - QUESTION 3
================================================================================
  Academic Performance Predictive Model using Deep Neural Network (DNN)
  Objective:
    Develop a Deep Neural Network (DNN) in TensorFlow to predict whether a
    student will PASS or FAIL based on their study hours and attendance percentage.
================================================================================
"""

import os
import sys

# Suppress TensorFlow informational logs for clean terminal output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models


def print_banner(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def generate_student_dataset(n_samples=1000, random_seed=42):
    """
    Generates a realistic synthetic academic dataset:
      - Study Hours: 0 to 14 hours per week
      - Attendance Percentage: 40% to 100%
      - Target: 1 (PASS) or 0 (FAIL) based on realistic academic criteria
    """
    np.random.seed(random_seed)
    
    # Study hours (normal distribution centered around 6.5 hours)
    study_hours = np.clip(np.random.normal(loc=6.5, scale=2.8, size=n_samples), 0.5, 14.0)
    
    # Attendance percentage (normal distribution centered around 76%)
    attendance = np.clip(np.random.normal(loc=76.0, scale=12.5, size=n_samples), 40.0, 100.0)
    
    # Academic score formula with realistic probabilistic boundary + slight noise
    # Passing requires balanced commitment in both classroom attendance and independent study
    composite_score = (0.40 * (study_hours / 14.0)) + (0.60 * (attendance / 100.0))
    noise = np.random.normal(loc=0.0, scale=0.05, size=n_samples)
    final_score = composite_score + noise
    
    # Pass threshold set at composite score >= 0.55
    labels = (final_score >= 0.55).astype(int)
    
    df = pd.DataFrame({
        "Study_Hours": np.round(study_hours, 2),
        "Attendance_Percentage": np.round(attendance, 2),
        "Result": labels
    })
    return df


def main():
    print_banner("STUDENT ACADEMIC PERFORMANCE PREDICTION USING TENSORFLOW DNN")
    print("Application Context:")
    print("  Predicting at-risk students early in an academic semester enables targeted")
    print("  mentorship and intervention. This DNN classifies student outcomes (Pass/Fail)")
    print("  using multi-layer non-linear combinations of Study Hours and Attendance.")

    # --------------------------------------------------------------------------
    # STEP 1: LOAD & PREPARE DATASET
    # --------------------------------------------------------------------------
    print_banner("STEP 1: DATASET GENERATION & PREPROCESSING")
    df = generate_student_dataset(n_samples=1000, random_seed=42)
    
    pass_count = (df["Result"] == 1).sum()
    fail_count = (df["Result"] == 0).sum()
    print(f"Total Student Records : {len(df)}")
    print(f"  - Passed Students (1): {pass_count} ({pass_count/len(df)*100:.1f}%)")
    print(f"  - Failed Students (0): {fail_count} ({fail_count/len(df)*100:.1f}%)\n")
    print("First 5 Student Records in Dataset:")
    print(df.head().to_string(index=False))

    X = df[["Study_Hours", "Attendance_Percentage"]].values
    y = df["Result"].values

    # Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nData Partitioning:")
    print(f"  - Training Samples : {len(X_train)}")
    print(f"  - Testing Samples  : {len(X_test)}")

    # Standardize Features (Zero mean, unit variance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Feature Scaling: Applied StandardScaler to Study Hours and Attendance.")

    # --------------------------------------------------------------------------
    # STEP 2: BUILD DEEP NEURAL NETWORK (DNN) ARCHITECTURE
    # --------------------------------------------------------------------------
    print_banner("STEP 2: TENSORFLOW DNN MODEL ARCHITECTURE")
    
    model = models.Sequential([
        layers.Input(shape=(2,), name="Input_Layer"),
        layers.Dense(32, activation="relu", name="Dense_Hidden_1"),
        layers.Dropout(0.2, name="Dropout_1"),
        layers.Dense(16, activation="relu", name="Dense_Hidden_2"),
        layers.Dense(8, activation="relu", name="Dense_Hidden_3"),
        layers.Dense(1, activation="sigmoid", name="Output_Classification")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # --------------------------------------------------------------------------
    # STEP 3: TRAIN THE DNN MODEL
    # --------------------------------------------------------------------------
    print_banner("STEP 3: TRAINING THE DNN MODEL")
    epochs = 30
    batch_size = 32
    print(f"Training for {epochs} epochs with batch size {batch_size}...")

    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )

    # --------------------------------------------------------------------------
    # STEP 4: EVALUATION ON UNSEEN TEST DATA
    # --------------------------------------------------------------------------
    print_banner("STEP 4: MODEL EVALUATION ON UNSEEN TEST SET")
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc * 100:.2f}%\n")

    # Generate Predictions & Evaluation Report
    y_pred_probs = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print(f"  True Negatives (Correct Fail) : {tn}")
    print(f"  False Positives (Type I Error): {fp}")
    print(f"  False Negatives (Type II Error): {fn}")
    print(f"  True Positives (Correct Pass) : {tp}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fail (0)", "Pass (1)"]))

    # --------------------------------------------------------------------------
    # STEP 5: PREDICT FOR NEW CANDIDATE STUDENTS
    # --------------------------------------------------------------------------
    print_banner("STEP 5: PREDICTING PASS / FAIL FOR NEW CANDIDATE STUDENTS")
    
    sample_students = [
        {"name": "Student A", "study_hours": 9.5, "attendance": 92.0, "scenario": "High study + High attendance"},
        {"name": "Student B", "study_hours": 1.5, "attendance": 50.0, "scenario": "Low study + Low attendance"},
        {"name": "Student C", "study_hours": 6.0, "attendance": 80.0, "scenario": "Moderate study + Good attendance"},
        {"name": "Student D", "study_hours": 3.0, "attendance": 62.0, "scenario": "Low-moderate borderline profile"},
        {"name": "Student E", "study_hours": 8.0, "attendance": 58.0, "scenario": "Compensatory: High study, lower attendance"}
    ]

    print(f"{'Student':<12} | {'Study Hrs':<10} | {'Attendance':<12} | {'Confidence':<12} | {'Prediction':<10} | {'Scenario'}")
    print("-" * 88)

    for student in sample_students:
        features = np.array([[student["study_hours"], student["attendance"]]])
        features_scaled = scaler.transform(features)
        prob = model.predict(features_scaled, verbose=0)[0][0]
        verdict = "PASS" if prob >= 0.5 else "FAIL"
        
        confidence = prob if verdict == "PASS" else (1.0 - prob)
        print(
            f"{student['name']:<12} | "
            f"{student['study_hours']:>9.1f}h | "
            f"{student['attendance']:>11.1f}% | "
            f"{confidence * 100:>10.2f}% | "
            f"{verdict:<10} | "
            f"{student['scenario']}"
        )

    print("\n" + "=" * 80)
    print("  DNN ACADEMIC PERFORMANCE PREDICTOR COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
