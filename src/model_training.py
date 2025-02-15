import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import pickle

# Load the dataset
def load_data(updated_file_path):
    return pd.read_csv(updated_file_path)

# Train the model
def train_model(df):
    # Print column names to verify the target column
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
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Impute missing values in both training and test data using the mean
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Model training complete. R^2 Score on Test Set: {score:.4f}")
    
    return model

if __name__ == "__main__":
    # Specify the path to the preprocessed dataset
    input_file = r"C:\Users\shrit\mlproject\cleaneddata\preprocessed_crime_dataset.csv"
    
    # Specify the path to save the trained model
    model_dir = r"C:\Users\shrit\mlproject\models"
    model_file = os.path.join(model_dir, "linear_regression_model.pkl")
    
    # Ensure the model directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Load the preprocessed dataset
    df = load_data(input_file)
    
    # Train the Linear Regression model
    model = train_model(df)
    
    # Save the trained model using pickle
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_file}.")

