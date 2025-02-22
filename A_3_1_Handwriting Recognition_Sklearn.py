#This python scrip shows that how 3 algorithm(SVM, LR, RF) used in training and testing the minst dataset and build a handwriting model.
#use Liner regression, svm, random forest to classify the hand-writing numbers in 10 input classes(0-9)
#visualizing the MSE error against Epoch for 3 algorithm in one line plot, with different colours for each algorithm.
#Author: TEJA SUDHASHREE DEVAGUPTAPU
#Date: 21-02-25
# A.3.1: Handwriting Recognition.

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target.astype(int)  # Convert labels to integers

# Use a subset: 6000 for training, 10000 for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=6000, test_size=10000, random_state=42, stratify=y
)

# Preprocess the data

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Initialize models and parameters

epochs = 50  # number of epochs for training

# Lists to store MSE for each epoch for each algorithm
mse_lr = []   # Linear Regression
mse_svm = []  # SVM with linear kernel
mse_rf = []   # Random Forest

# Linear Regression (using SGDRegressor with squared_error loss)
lr_model = SGDRegressor(loss='squared_error', max_iter=1, tol=None,
                        learning_rate='constant', eta0=0.0001, alpha=0.0001, random_state=42, warm_start=True)

# SVM with linear kernel (using SGDClassifier with hinge loss)
svm_model = SGDClassifier(loss='hinge', max_iter=1, tol=None,
                          learning_rate='constant', eta0=0.01, random_state=42, warm_start=True)

# Random Forest Classifier (we enable warm_start so we can incrementally add trees)
rf_model = RandomForestClassifier(n_estimators=5, max_depth=10, warm_start=True, random_state=42)

# For the SGD models, perform an initial partial fit.
# (For SGDClassifier, provide the list of possible classes)
lr_model.partial_fit(X_train, y_train)
svm_model.partial_fit(X_train, y_train, classes=np.unique(y_train))

# Training loop over epochs
for epoch in range(epochs):
    # ----- Linear Regression -----
    lr_model.partial_fit(X_train, y_train)
    # SGDRegressor returns continuous values; round them to the nearest integer (0-9)
    y_pred_lr = lr_model.predict(X_train)
    y_pred_lr_rounded = np.rint(y_pred_lr)
    mse_lr_epoch = mean_squared_error(y_train, y_pred_lr_rounded)
    mse_lr.append(mse_lr_epoch)

    # ----- SVM (Linear) -----
    svm_model.partial_fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_train)
    mse_svm_epoch = mean_squared_error(y_train, y_pred_svm)
    mse_svm.append(mse_svm_epoch)

    # ----- Random Forest -----
    # Increase the number of trees by 5 each epoch and refit.
    rf_model.n_estimators += 5
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_train)
    mse_rf_epoch = mean_squared_error(y_train, y_pred_rf)
    mse_rf.append(mse_rf_epoch)

    print(f"Epoch {epoch+1:02d}: LR MSE = {mse_lr_epoch:.3f}, SVM MSE = {mse_svm_epoch:.3f}, RF MSE = {mse_rf_epoch:.3f}")

# Plot MSE vs Epoch for all three algorithms
plt.figure(figsize=(10, 6))
epochs_range = np.arange(1, epochs + 1)
plt.plot(epochs_range, mse_lr, label="Linear Regression", color='blue')
plt.plot(epochs_range, mse_svm, label="SVM", color='red')
plt.plot(epochs_range, mse_rf, label="Random Forest", color='green')
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("MSE Error")
plt.title("MSE Error vs Epoch for LR, SVM, and RF")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()