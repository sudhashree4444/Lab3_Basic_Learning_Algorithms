#This Python script shows training a ML model to predict which direction the arduino moves towards.
# collecting 40 data records with low frequency 100hz and duration of 1 second each, with 20 right and 20 left.
#Exporting the data as JSON, the script here transforms them to CSV.
#Author: TEJA SUDHASHREE DEVAGUPTAPU
#Date: 21-02-25
# A.3.3: Guess where did I GO! (Arduino Inertia)

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Function given in Assignment to load JSON files (original function)
def load_json_data(json_files):
    data = []
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)  # Load JSON file
            data.append(json_data)  # Append to list
    return data

# Function to process JSON data into structured arrays
def process_json_data(json_data):
    X = []
    y = []

    for records in json_data: # to loop all records in file 40 records
        for record in records:  # Loop to all sub-elements in each record
            acc_x = np.mean([sample["ax"] for sample in record["samples"]])
            acc_y = np.mean([sample["ay"] for sample in record["samples"]])
            acc_z = np.mean([sample["az"] for sample in record["samples"]])

            direction = record["direction"]  # Right or Left

            X.append([acc_x, acc_y, acc_z])
            y.append(direction)

    return np.array(X), np.array(y)

# give Path to Json file here, give comma separated if more than one file
json_files = ["data/Arduino_Inertia1.json"]

# Load JSON data (raw)
json_data = load_json_data(json_files)

# Process the loaded data into lists
X, y = process_json_data(json_data)

# Convert labels ('left'/'right') into numerical values (0 or 1) since Scikit-learn models need numeric labels
label_encoder = LabelEncoder()
y_boolean = label_encoder.fit_transform(y)  # "Left" -> 0, "Right" -> 1

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y_boolean, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and round to nearest integer (0 or 1)
y_pred = np.round(model.predict(X_test)).astype(int)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100}%")

# Export data to CSV
df = pd.DataFrame(X, columns=['ACC_X', 'ACC_Y', 'ACC_Z'])
df['Label'] = y
csv_filename = "data/arduino_movement_data.csv"
df.to_csv(csv_filename, index=False)
print(f"Data saved to '{csv_filename}'.")

