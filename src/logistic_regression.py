import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.impute import SimpleImputer

def load_data(updated_file_path):
    return pd.read_csv(updated_file_path)

def train_logistic_model(df):
    # Print column names to verify the target column
    print("Available columns in the dataset:", df.columns)
    
    # Set the correct target column name
    target_column = 'Crime Rate'  # Ensure this matches the actual target column name
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    # Binarize the target column for classification (e.g., high vs low crime rate)
    df['CRIME_TARGET'] = (df[target_column] > df[target_column].median()).astype(int)  # 1 for above median, 0 for below

    # Features and target variable
    X = df.drop([target_column, 'CRIME_TARGET'], axis=1)
    y = df['CRIME_TARGET']
    
    # If there are non-numeric columns, convert them to numeric
    X = pd.get_dummies(X)  # Convert categorical variables into dummy/indicator variables
    
    # Handle NaN values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)  # Impute missing values in features
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Evaluate the model: accuracy, precision, recall, f1 score, and ROC-AUC
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Print classification performance metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Plot confusion matrix and ROC curve
    plt.figure(figsize=(12, 6))

    # Confusion matrix plot
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Crime', 'High Crime'], 
                yticklabels=['Low Crime', 'High Crime'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # ROC Curve plot
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.tight_layout()  # Adjust spacing
    plt.show()
    
    return model

if __name__ == "__main__":
    # Specify the path to the preprocessed dataset
    input_file = r"cleaneddata\preprocessed_crime_dataset.csv"
    
    # Specify the path to save the trained model
    model_dir = r"C:\Users\shrit\mlproject\models"
    model_file = os.path.join(model_dir, "logistic_regression_model.pkl")
    
    # Ensure the model directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Load the preprocessed dataset
    df = load_data(input_file)
    
    # Train the Logistic Regression model
    model = train_logistic_model(df)
    
    # Save the trained model
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_file}.")
