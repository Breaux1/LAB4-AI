# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset
# Replace with the path to your dataset file
dataset_file = "Dictionary Brute Force.csv"
try:
    df = pd.read_csv(dataset_file)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: The dataset file '{dataset_file}' was not found. Please check the file path and try again.")
    exit()

# Display the first few rows of the dataset
print(df.head())

# Step 2: Data Preprocessing
# Check for missing values and handle them
print("\nChecking for missing values...")
print(df.isnull().sum())

# Identify numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Optionally, handle non-numeric columns if necessary
# For example, you can fill missing values with the mode or a placeholder
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')

# Select relevant features for anomaly detection
# Modify this based on the actual column names in the dataset
features = ["duration", "protocol_type", "bytes", "packets"]

# Ensure that selected features are present in the DataFrame
if not all(feature in df.columns for feature in features):
    print(f"Error: One or more features {features} not found in the dataset. Please check the column names.")
    exit()

# Check if selected features are numeric where necessary
for feature in ["duration", "bytes", "packets"]:
    if not pd.api.types.is_numeric_dtype(df[feature]):
        print(f"Error: Feature '{feature}' is not numeric. Please ensure all selected features are numeric.")
        exit()

df_selected = df[features]

# One-hot encode categorical features (e.g., protocol_type)
df_encoded = pd.get_dummies(df_selected, columns=["protocol_type"], drop_first=True)

# Normalize data to bring all features to the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# Step 3: Apply K-Means Clustering
print("\nApplying K-Means Clustering...")
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
df["cluster"] = kmeans.labels_

# Flag anomalies based on distance to cluster centroids
distances = kmeans.transform(X_scaled).min(axis=1)
threshold = np.percentile(distances, 95)  # Top 5% as anomalies
df["anomaly_kmeans"] = distances > threshold

# Step 4: Apply DBSCAN
print("\nApplying DBSCAN...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
df["anomaly_dbscan"] = dbscan_labels == -1  # Noise points are labeled as -1 in DBSCAN

# Step 5: PCA for Visualization
print("\nApplying PCA for Dimensionality Reduction...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize PCA results with K-Means anomalies
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["anomaly_kmeans"], cmap="coolwarm", label="Anomaly (K-Means)")
plt.title("PCA Visualization of Anomalies (K-Means)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Anomaly (1 = Anomaly, 0 = Normal)")
plt.legend()
plt.show()

# Visualize PCA results with DBSCAN anomalies
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["anomaly_dbscan"], cmap="coolwarm", label="Anomaly (DBSCAN)")
plt.title("PCA Visualization of Anomalies (DBSCAN)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Anomaly (1 = Anomaly, 0 = Normal)")
plt.legend()
plt.show()

# Step 6: Apply Isolation Forest (Optional)
print("\nApplying Isolation Forest...")
isolation_forest = IsolationForest(random_state=42, contamination=0.05)  # 5% contamination
df["anomaly_isolation_forest"] = isolation_forest.fit_predict(X_scaled)
df["anomaly_isolation_forest"] = df["anomaly_isolation_forest"].apply(lambda x: 1 if x == -1 else 0)

# Visualize PCA results with Isolation Forest anomalies
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["anomaly_isolation_forest"], cmap="coolwarm", label="Anomaly (Isolation Forest)")
plt.title("PCA Visualization of Anomalies (Isolation Forest)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Anomaly (1 = Anomaly, 0 = Normal)")
plt.legend()
plt.show()

# Step 7: Evaluate Results (Optional - if labels are available)
# If labeled data is available, calculate evaluation metrics like Precision, Recall, and F1-Score
if "true_labels" in df.columns:  # Replace 'true_labels' with the actual label column in your dataset
    from sklearn.metrics import classification_report, confusion_matrix

    print("\nEvaluation of K-Means Anomalies:")
    print(classification_report(df["true_labels"], df["anomaly_kmeans"]))

    print("\nEvaluation of DBSCAN Anomalies:")
    print(classification_report(df["true_labels"], df["anomaly_dbscan"]))

    print("\nEvaluation of Isolation Forest Anomalies:")
    print(classification_report(df["true_labels"], df["anomaly_isolation_forest"]))

# Final Step: Save the results to a CSV file
output_file = "anomaly_detection_results.csv"
df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")