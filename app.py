# ============================================================
# BREAST CANCER CLASSIFICATION
# Breast Cancer Wisconsin (Diagnostic) Dataset
#
# Models:
# 1. Logistic Regression
# 2. Support Vector Machine (SVM)
# 3. K-Nearest Neighbors (KNN)
# 4. Random Forest
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report

# ------------------------------------------------------------
# 2. LOAD THE BREAST CANCER WISCONSIN DATASET
# ------------------------------------------------------------

data = load_breast_cancer()

# Features
X =data.data

# Target
y = data.target

print("Dataset loaded successfully!")

print("\nNumber of rows:", X.shape[0])
print("Number of features:", X.shape[1])



# ------------------------------------------------------------
# 3. SPLIT THE DATASET INTO TRAINING AND TESTING DATA
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)


# ------------------------------------------------------------
# 4. FEATURE SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")


# ============================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================

print("\n================================================")
print("MODEL 1: LOGISTIC REGRESSION")
print("================================================")


# Create the model
logistic_model = LogisticRegression(max_iter=4000)


# Train and fit the model
logistic_model.fit(X_train_scaled, y_train)

# Make predictions
logistic_prediction = logistic_model.predict(X_test_scaled)


# Calculate performance
logistic_accuracy = accuracy_score(y_test, logistic_prediction)
logistic_precision = precision_score(y_test, logistic_prediction)
logistic_recall = recall_score(y_test, logistic_prediction)
logistic_f1 = f1_score(y_test, logistic_prediction)


print("\nPredictions:")
print(logistic_prediction)

print("\nClassification Report:")
print(classification_report(
    y_test,
    logistic_prediction,
    target_names=data.target_names
))

print("Logistic Regression Accuracy:",
      round(logistic_accuracy * 100, 2), "%")


# ============================================================
# MODEL 2: SUPPORT VECTOR MACHINE
# ============================================================

print("\n================================================")
print("MODEL 2: SUPPORT VECTOR MACHINE")
print("================================================")


# Create the model
svm_model = SVC(kernel="linear")


# Train and fit the model
svm_model.fit(X_train_scaled, y_train)


# Make predictions
svm_prediction = svm_model.predict(X_test_scaled)


# Calculate performance
svm_accuracy = accuracy_score(y_test, svm_prediction)
svm_precision = precision_score(y_test, svm_prediction)
svm_recall = recall_score(y_test, svm_prediction)
svm_f1 = f1_score(y_test, svm_prediction)


print("\nPredictions:")
print(svm_prediction)

print("\nClassification Report:")
print(classification_report(
    y_test,
    svm_prediction,
    target_names=data.target_names
))

print("SVM Accuracy:",
      round(svm_accuracy * 100, 2), "%")


# ============================================================
# MODEL 3: K-NEAREST NEIGHBORS
# ============================================================

print("\n================================================")
print("MODEL 3: K-NEAREST NEIGHBORS")
print("================================================")


# Create the model
knn_model = KNeighborsClassifier(n_neighbors=6)


# Train and fit the model
knn_model.fit(X_train_scaled, y_train)


# Make predictions
knn_prediction = knn_model.predict(X_test_scaled)


# Calculate performance
knn_accuracy = accuracy_score(y_test, knn_prediction)
knn_precision = precision_score(y_test, knn_prediction)
knn_recall = recall_score(y_test, knn_prediction)
knn_f1 = f1_score(y_test, knn_prediction)


print("\nPredictions:")
print(knn_prediction)

print("\nClassification Report:")
print(classification_report(
    y_test,
    knn_prediction,
    target_names=data.target_names
))

print("KNN Accuracy:",
      round(knn_accuracy * 100, 2), "%")


# ============================================================
# MODEL 4: RANDOM FOREST
# ============================================================

print("\n================================================")
print("MODEL 4: RANDOM FOREST")
print("================================================")


# Create the model
random_forest_model = RandomForestClassifier(
    n_estimators=98,
    random_state=42
)


# Train and fit the model
random_forest_model.fit(X_train, y_train)


# Make predictions
random_forest_prediction = random_forest_model.predict(X_test)


# Calculate performance
random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_prediction
)

random_forest_precision = precision_score(
    y_test,
    random_forest_prediction
)

random_forest_recall = recall_score(
    y_test,
    random_forest_prediction
)

random_forest_f1 = f1_score(
    y_test,
    random_forest_prediction
)


print("\nPredictions:")
print(random_forest_prediction)

print("\nClassification Report:")
print(classification_report(
    y_test,
    random_forest_prediction,
    target_names=data.target_names
))

print("Random Forest Accuracy:",
      round(random_forest_accuracy * 100, 2), "%")


# ============================================================
# 5. FINAL COMPARISON TABLE
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "SVM",
        "KNN",
        "Random Forest"
    ],

    "Accuracy": [
        logistic_accuracy,
        svm_accuracy,
        knn_accuracy,
        random_forest_accuracy
    ],

    "Precision": [
        logistic_precision,
        svm_precision,
        knn_precision,
        random_forest_precision
    ],

    "Recall": [
        logistic_recall,
        svm_recall,
        knn_recall,
        random_forest_recall
    ],

    "F1 Score": [
        logistic_f1,
        svm_f1,
        knn_f1,
        random_forest_f1
    ]
})


# Convert values into percentages
results["Accuracy"] = results["Accuracy"] * 100
results["Precision"] = results["Precision"] * 100
results["Recall"] = results["Recall"] * 100
results["F1 Score"] = results["F1 Score"] * 100


print("\n\n================================================")
print("FINAL MODEL COMPARISON")
print("================================================")

print(
    results.round(2).to_string(index=False)
)


# ============================================================
# 6. BAR CHART FOR MODEL COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

models = results["Model"]
accuracy_values = results["Accuracy"]

plt.bar(
    models,
    accuracy_values,
    color=["lightgreen", "pink", "lightblue", "violet"]
)

plt.title("Breast Cancer Classification - Model Accuracy")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)


plt.show()



# ============================================================
# 7. PREDICT NEW SAMPLES
# ============================================================

# New sample with 30 input feature values
new_sample = [[
    14.0, 20.0, 90.0, 600.0, 0.10,
    0.10, 0.08, 0.05, 0.18, 0.06,
    0.40, 1.20, 2.50, 40.0, 0.006,
    0.02, 0.025, 0.01, 0.02, 0.003,
    16.0, 25.0, 105.0, 800.0, 0.13,
    0.25, 0.20, 0.08, 0.28, 0.07
]]

# Convert new sample into a DataFrame with feature names
new_sample_df = pd.DataFrame(
    new_sample
)

# Scale the sample for Logistic Regression, SVM and KNN
new_sample_scaled = scaler.transform(new_sample_df)

# Predict using all four trained models
lr_result = logistic_model.predict(new_sample_scaled)
svm_result = svm_model.predict(new_sample_scaled)
knn_result = knn_model.predict(new_sample_scaled)

# Random Forest was trained using unscaled features
rf_result = random_forest_model.predict(new_sample_df)

# Display predictions
print("\n====================================")
print("NEW SAMPLE PREDICTION")
print("====================================")

print("Logistic Regression:",
      int(lr_result[0]))

print("SVM:",
      int(svm_result[0]))

print("KNN:",
      int(knn_result[0]))

print("Random Forest:",
      int(rf_result[0]))

print("====================================")