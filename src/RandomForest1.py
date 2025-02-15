# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv(r'cleaneddata\preprocessed_crime_dataset.csv')

# Drop columns that have all NaN values
df = df.dropna(axis=1, how='all')

# Separate numeric and non-numeric columns after dropping columns
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

# Train and evaluate Naive Bayes model
nb = GaussianNB()
nb.fit(X_train_pca, y_train)
y_pred_nb = nb.predict(X_test_pca)

# Naive Bayes model evaluation
accuracy = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Accuracy:", accuracy)
print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))
print("Naive Bayes Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

# Plot Naive Bayes predictions on test data
plt.figure(figsize=(8, 6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred_nb, cmap='viridis', edgecolors='k')
plt.title("Naive Bayes Predictions on Test Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Predicted Crime Rate Category")
plt.show()

# Train a Random Forest on the PCA-transformed data
rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=2)  # Set max_depth=2 for simplicity
rf.fit(X_train_pca, y_train)

# Random Forest predictions and evaluation
y_pred_rf = rf.predict(X_test_pca)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Visualize feature importances
importances = rf.feature_importances_
features = ["Principal Component 1", "Principal Component 2"]
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Feature Importances in Random Forest Model (PCA-Transformed Data)")
plt.show()

# Visualize the first two individual trees in the Random Forest
for i in range(2):  # Adjust the range if you want to visualize more trees
    plt.figure(figsize=(12, 8))
    plot_tree(rf.estimators_[i], feature_names=["Principal Component 1", "Principal Component 2"], 
              class_names=["Low", "High"], filled=True, rounded=True)
    plt.title(f"Random Forest Tree {i+1} Visualization (Depth=2)")
    plt.show()

# Visualize Random Forest predictions on test data
plt.figure(figsize=(8, 6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred_rf, cmap='viridis', edgecolors='k')
plt.title("Random Forest Predictions on Test Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Predicted Crime Rate Category")
plt.show()
