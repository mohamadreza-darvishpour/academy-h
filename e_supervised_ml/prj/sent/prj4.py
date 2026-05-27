# ---------------------------------------------------
# Import required libraries
# ---------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Dataset and train-test split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# Machine learning models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Evaluation metric
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# Load Olivetti Faces dataset
# ---------------------------------------------------

faces = fetch_olivetti_faces()

# X = images data
# Shape -> (400, 64, 64)
X = faces.images

# y = target labels (person IDs)
y = faces.target

# ---------------------------------------------------
# Print number of persons
# ---------------------------------------------------

unique_persons = np.unique(y)

print("Number of persons in dataset:", len(unique_persons))

# ---------------------------------------------------
# Display one image from each person
# ---------------------------------------------------

plt.figure(figsize=(15, 8))

# Loop through all persons
for i, person_id in enumerate(unique_persons):

    # Find first image index of each person
    index = np.where(y == person_id)[0][0]

    # Create subplot
    plt.subplot(4, 10, i + 1)

    # Display image
    plt.imshow(X[index], cmap='gray')

    # Image title
    plt.title(f"P{person_id}")

    # Hide axes
    plt.axis('off')

# Adjust spacing
plt.tight_layout()

# Show images together
plt.show()

# ---------------------------------------------------
# Prepare data for machine learning
# Convert 64x64 images to 1D vectors
# ---------------------------------------------------

X = X.reshape((X.shape[0], -1))

# ---------------------------------------------------
# Split dataset into training and testing sets
# 80% train and 20% test
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===================================================
# Part (B) - SVM Model
# ===================================================

# Create SVM classifier
svm_model = SVC(kernel='linear')

# Train model
svm_model.fit(X_train, y_train)

# Predict test data
y_pred_svm = svm_model.predict(X_test)

# Calculate accuracy
svm_accuracy = accuracy_score(y_test, y_pred_svm)

print("\nSVM Accuracy:", svm_accuracy)

# ===================================================
# Part (C) - Random Forest Model
# ===================================================

# Create Random Forest classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

# Predict test data
y_pred_rf = rf_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("Random Forest Accuracy:", rf_accuracy)

# ===================================================
# Part (C) - AdaBoost Model
# ===================================================

# Create AdaBoost classifier
ada_model = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
ada_model.fit(X_train, y_train)

# Predict test data
y_pred_ada = ada_model.predict(X_test)

# Calculate accuracy
ada_accuracy = accuracy_score(y_test, y_pred_ada)

print("AdaBoost Accuracy:", ada_accuracy)

# ===================================================
# Compare models
# ===================================================

print("\n------ Model Comparison ------")
print(f"SVM Accuracy           : {svm_accuracy:.4f}")
print(f"Random Forest Accuracy : {rf_accuracy:.4f}")
print(f"AdaBoost Accuracy      : {ada_accuracy:.4f}")

# Find best model
best_accuracy = max(svm_accuracy, rf_accuracy, ada_accuracy)

if best_accuracy == svm_accuracy:
    print("\nBest Model: SVM")

elif best_accuracy == rf_accuracy:
    print("\nBest Model: Random Forest")

else:
    print("\nBest Model: AdaBoost")