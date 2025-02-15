import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the dataset
def load_data(updated_file_path):
    return pd.read_csv(updated_file_path)

# Train the model
def train_model(df):
    print("Available columns in the dataset:", df.columns)
    
    # Set the correct target column name (update this if necessary)
    target_column = 'Crime Rate'  # Replace with the actual target column
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    # Features and target variable
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Remove columns with all missing values
    X = X.dropna(axis=1, how='all')
    
    # Convert categorical variables into dummy/indicator variables (one-hot encoding)
    X = pd.get_dummies(X, drop_first=True)

    # Impute missing values in both training and test data using the mean
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)  # Fit and transform for both training and test set

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Model training complete. R^2 Score on Test Set: {score:.4f}")
    
    return model, X_train, y_train, X_test, y_test

# Visualization function
def visualize_results(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Create a DataFrame for visualization
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
    # Set the style
    sns.set(style='whitegrid')

    # Create a scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(results['Actual'], results['Predicted'], 
                          alpha=0.6, c='blue', s=100, edgecolor='k', label='Predicted Points')

    # Add a red line for perfect predictions
    plt.plot([results['Actual'].min(), results['Actual'].max()], 
             [results['Actual'].min(), results['Actual'].max()], 
             color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

    # Add title and labels
    plt.title('Actual vs Predicted Values', fontsize=16)
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    
    # Set x and y limits to better focus on the data
    plt.xlim(results['Actual'].min() - 1, results['Actual'].max() + 1)
    plt.ylim(results['Predicted'].min() - 1, results['Predicted'].max() + 1)

    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Specify the path to the preprocessed dataset
    input_file = r'cleaneddata\preprocessed_crime_dataset.csv'
    
    # Load the preprocessed dataset
    df = load_data(input_file)
    
    # Train the Linear Regression model
    model, X_train, y_train, X_test, y_test = train_model(df)
    
    # Visualize the results
    visualize_results(model, X_test, y_test)
    
    # Specify the path to save the trained model
    model_dir = r"C:\Users\shrit\mlproject\models"
    model_file = os.path.join(model_dir, "linear_regression_model.pkl")
    
    # Ensure the model directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save the trained model using pickle
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_file}.")

import matplotlib.pyplot as plt
import seaborn as sns

