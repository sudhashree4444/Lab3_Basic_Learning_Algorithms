#This Python script shows the dataset that is the classification of 3 types of penguins based on the length of their bill(beak).
#Load the dataset
#build a k-means clustring model to cluster the penguins.
#Evaluating the model and finding the accuracy.
#Author: TEJA SUDHASHREE DEVAGUPTAPU
#Date: 22-02-25
# A.3.4: Classify the Pinguins (Unsupervised).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# Load the dataset
file_path = "data/penguins.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Select relevant columns and drop missing values
df_filtered = df[['species', 'bill_length_mm', 'bill_depth_mm']].dropna()

# Extract features and labels
X = df_filtered[['bill_length_mm', 'bill_depth_mm']].values
y_true = df_filtered['species'].values

# Convert species names to numeric labels
label_encoder = LabelEncoder()
y_true_encoded = label_encoder.fit_transform(y_true)

# Apply K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Match clusters to true labels
labels_map = np.zeros_like(y_kmeans)
for i in range(3):
    mask = (y_kmeans == i)
    labels_map[mask] = mode(y_true_encoded[mask])[0]

# Calculate accuracy
accuracy = accuracy_score(y_true_encoded, labels_map)

# Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6, edgecolors='k', label="Clustered Points")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label="Centroids")

plt.xlabel("Bill Length (mm)")
plt.ylabel("Bill Depth (mm)")
plt.title(f"K-Means Clustering of Penguins (Accuracy: {accuracy * 100 :.2f}%)")
plt.legend()
plt.show()

print(f"Model Accuracy: {accuracy * 100:.2f}%")
