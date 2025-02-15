# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv(r'cleaneddata/preprocessed_crime_dataset.csv')

# Drop columns that have all NaN values
df = df.dropna(axis=1, how='all')

# Separate numeric and non-numeric columns again after dropping columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Handle missing values
# Impute numeric columns with the mean
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

# Impute categorical columns with the most frequent value
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

# Convert the 'City' column to categorical labels using LabelEncoder
le = LabelEncoder()
df['City'] = le.fit_transform(df['City'])

# Define the features (X) for clustering
X = df[['Personal Revenge', 'Anger', 'Fraud', 'Extortion', 'Causing Disrepute', 'Prank', 'Sexual Exploitation',
        'Political Motives', 'Terrorist Activities', 'Inciting Hate against Country', 'Disrupt Public Service',
        'Spreading Piracy', 'Others', 'Total']]

# Split the data into training and testing sets (if needed, but not necessary for clustering)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Perform PCA on the scaled data (reduce to 2 principal components for visualization)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# Elbow method to find the optimal number of clusters
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_pca)
    inertia.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.title("Elbow Method for Optimal k")
plt.show()

# Choose the optimal number of clusters (e.g., if elbow is around k=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_train_pca)
clusters = kmeans.labels_

# Plot the PCA-transformed data with K-Means clusters and centers
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=clusters, cmap='viridis')
plt.colorbar(label='Cluster', ticks=range(optimal_k))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', label='Cluster Centers', marker='X')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'K-Means Clustering with {optimal_k} Centers')
plt.legend()
plt.show()
