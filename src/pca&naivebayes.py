# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv(r'cleaneddata\preprocessed_crime_dataset.csv')

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

# Define the features (X) and the target (y)
X = df[['Personal Revenge', 'Anger', 'Fraud', 'Extortion', 'Causing Disrepute', 'Prank', 'Sexual Exploitation',
        'Political Motives', 'Terrorist Activities', 'Inciting Hate against Country', 'Disrupt Public Service',
        'Spreading Piracy', 'Others', 'Total']]

# Convert the continuous 'Crime Rate' to categorical by binning into low and high
df['Crime Rate Category'] = pd.cut(df['Crime Rate'], bins=2, labels=[0, 1])

# Define target variable (Crime Rate Category)
y = df['Crime Rate Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform PCA on the scaled data (reduce to 2 principal components)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Perform Naive Bayes on the PCA-transformed data
nb = GaussianNB()
nb.fit(X_train_pca, y_train)

# Make predictions on the test data
y_pred_nb = nb.predict(X_test_pca)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_nb)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred_nb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

# Plot the PCA-transformed training data
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolors='k')
plt.title("PCA-Transformed Training Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Crime Rate Category")
plt.show()

# Plot the Naive Bayes predictions on test data
plt.figure(figsize=(8, 6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred_nb, cmap='viridis', edgecolors='k')
plt.title("Naive Bayes Predictions on Test Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Predicted Crime Rate Category")
plt.show()
