import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Step 1: Load the Cleaned Dataset
df_tracks = pd.read_csv('tracks_dataset_cleaned.csv')

# Step 2: Prepare Features and Target
# Define features and target column
# Drop unnecessary columns and convert categorical columns to numerical using one-hot encoding
features = df_tracks.drop(columns=['track_id', 'track_name', 'artist_name', 'lyrics'])
categorical_columns = features.select_dtypes(include=['object']).columns
features = pd.get_dummies(features, columns=categorical_columns, drop_first=True)

if 'mode' in df_tracks.columns:
    df_tracks['mode'] = df_tracks['mode'].astype('category')  # Ensure 'mode' is of categorical type
    target = df_tracks['mode'].cat.codes  # Encode the target
else:
    target = None

# Ensure target column is not None
if target is None:
    raise ValueError("Target column ('mode') not found in dataset")

# Step 3: Handle Class Imbalance
smote = SMOTE(random_state=42)
features_resampled, target_resampled = smote.fit_resample(features, target)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features_resampled, target_resampled, test_size=0.2, random_state=42)

# Step 5: Normalize Numerical Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# Step 7: Evaluate the Model with Cross-Validation
cross_val_scores = cross_val_score(model, features_resampled, target_resampled, cv=5)
print("Cross-Validation Scores:", cross_val_scores)
print("Mean Cross-Validation Score:", cross_val_scores.mean())

# Evaluate on Test Set
y_pred = model.predict(X_test)
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("Classification Report on Test Set:\n", classification_report(y_test, y_pred))

# Step 8: Save the Model
joblib.dump(model, 'music_mode_classifier.pkl')

# Step 9: Making Predictions
# Load the model and make predictions on new data
model = joblib.load('music_mode_classifier.pkl')
new_data = X_test[:5]  # Use a few rows from X_test for demo predictions
predictions = model.predict(new_data)
print("Predictions for new data:", predictions)

# Step 10: API Creation with Flask (Optional)
from flask import Flask, request, jsonify
import platform

app = Flask(__name__)

# Load the trained model
model = joblib.load('music_mode_xgb_classifier.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)  # Ensure the input matches training features
    df = scaler.transform(df)  # Normalize input data
    predictions = model.predict(df)
    return jsonify(predictions.tolist())

import subprocess

if __name__ == '__main__':
    # Use Waitress for Windows deployment, or Gunicorn via Docker
    if platform.system() == "Windows":
        from waitress import serve
        serve(app, host='0.0.0.0', port=5000)
    else:
        # Assuming Docker deployment for non-Windows systems
        subprocess.run(["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "ml2:app"])
