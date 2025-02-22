#This Python script shows, for given data set(weather conditions) we are trying to predict next day's weather.
#finding out how the weatheer conditions is going to be based on:There are 5 output classes: (1)drizzle, (2)rain, (3)sun, (4)snow, (5)fog
#Using  SVM (with Linear kernel), and Random Forest(with a maximum depth of less than 10) algorithms to classify the weather data in 5 output classes: "drizzle", "rain", "sun", "snow", "fog"
#Visualizing the MSE error against Epoch for 3 algorithms in one line plot, with different colors for each algorithm.
#Visualizing the results of one of the algorithms.
#Author: TEJA SUDHASHREE DEVAGUPTAPU
#Date: 20-02-25
# A.3.2:  Predict the Rain!  - IOT DATA.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = "data/seattle-weather.csv"
df = pd.read_csv(file_path)

# Display the first few rows
df.head()

# Drop the date column since it's not needed
df = df.drop(columns=['date'])

# Encode the target variable (weather) into numerical values
label_encoder = LabelEncoder()
df['weather'] = label_encoder.fit_transform(df['weather'])

# Define features and target variable
X = df.drop(columns=['weather'])
y = df['weather']

# Standardize the feature variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
#lr = LogisticRegression(max_iter=1000)  # Using Logistic Regression instead of Linear Regression for classification
linear_reg = LinearRegression()
svm = SVC(kernel='linear', probability=True)
rf = RandomForestClassifier(max_depth=9, random_state=42)

# Lists to store MSE for each epoch
mse_linear, mse_svm, mse_rf = [], [], []

# Train models and record MSE
epochs = np.arange(1, 11)

for epoch in epochs:
    # Logistic Regression
    # lr.fit(X_train[:epoch * 10], y_train[:epoch * 10])
    #y_pred_lr = lr.predict(X_test)
    #mse_lr.append(mean_squared_error(y_test, y_pred_lr))
    # Linear Regression
    linear_reg.fit(X_train[:epoch * 10], y_train[:epoch * 10])
    y_pred_linear = np.round(linear_reg.predict(X_test))  # Rounding predictions to nearest class
    y_pred_linear = np.clip(y_pred_linear, 0, 4)  # Ensuring predictions stay within valid class range
    mse_linear.append(mean_squared_error(y_test, y_pred_linear))

    # SVM
    svm.fit(X_train[:epoch * 10], y_train[:epoch * 10])
    y_pred_svm = svm.predict(X_test)
    mse_svm.append(mean_squared_error(y_test, y_pred_svm))

    # Random Forest
    rf.fit(X_train[:epoch * 10], y_train[:epoch * 10])
    y_pred_rf = rf.predict(X_test)
    mse_rf.append(mean_squared_error(y_test, y_pred_rf))

# Plot MSE against epochs
plt.figure(figsize=(8, 5))
plt.plot(epochs, mse_linear, marker='o', label='Linear Regression', color='red')
plt.plot(epochs, mse_svm, marker='s', label='SVM', color='blue')
plt.plot(epochs, mse_rf, marker='^', label='Random Forest', color='green')
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("MSE vs Epochs for Different Models")
plt.legend(loc='upper right')
plt.show()

# Confusion Matrix for Random Forest
y_pred_rf_final = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf_final)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Random Forest")
plt.show()