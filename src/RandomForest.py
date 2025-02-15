import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load and prepare the data
data = pd.DataFrame([
    ["Agra",5,0,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,46,70,4],
    ["Allahabad",0,0,222,11,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,241,19.8],
    ["Amritsar",2,0,5,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0.8],
    ["Asansol",6,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,21,1.7],
    ["Aurangabad",5,2,51,0,0,0,21,0,0,0,0,3,0,0,0,0,0,0,0,0,82,6.9],
    ["Bhopal",0,0,4,7,2,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,16,0.8],
    ["Chandigarh City",0,0,19,3,0,7,0,0,0,0,0,0,0,0,0,0,0,0,0,1,30,2.9],
    ["Dhanbad",2,0,29,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,2.6],
    ["Durg-Bhilainagar",0,0,0,0,10,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,0.9],
    ["Faridabad",0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,17,1.2],
    ["Gwalior",0,2,24,1,9,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,41,3.7],
    ["Jabalpur",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ["Jamshedpur",0,0,47,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,47,3.5],
    ["Jodhpur",0,0,59,0,4,0,6,0,0,0,0,0,0,0,0,0,0,0,0,42,111,9.8],
    ["Kannur",3,1,1,0,0,0,1,3,0,0,0,0,0,0,0,0,0,0,0,1,10,0.6],
    ["Kollam",0,0,7,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,12,1.1],
    ["Kota",0,3,6,5,5,3,3,0,0,0,0,1,0,0,0,0,0,0,0,0,26,2.6],
    ["Ludhiana",0,0,6,4,0,0,13,1,0,0,0,0,0,0,0,0,0,0,0,0,24,1.5],
    ["Madurai",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ["Malappuram",2,1,2,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,7,0.4],
    ["Meerut",0,0,0,0,53,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,64,4.5],
    ["Nasik",0,0,21,0,8,0,12,0,0,0,0,0,0,0,0,0,0,0,0,0,41,2.6],
    ["Raipur",0,0,3,0,8,0,9,1,0,0,0,0,0,0,0,0,0,0,0,2,23,2],
    ["Rajkot",0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,7,0.5],
    ["Ranchi",0,0,175,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,175,15.5],
    ["Srinagar",1,0,6,1,5,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,15,1.2],
    ["Thiruvananthapuram",7,2,23,0,2,1,2,3,3,0,0,0,0,0,0,0,0,0,0,4,47,2.8],
    ["Thrissur",0,0,3,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0.3],
    ["Tiruchirapalli",1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,4,6,0.6],
    ["Vadodara",0,0,11,0,1,1,0,0,0,0,0,2,0,0,0,0,0,0,0,0,15,0.8],
    ["Varanasi",12,14,111,5,17,31,31,2,0,0,0,2,5,0,0,8,0,0,0,49,287,20],
    ["Vasai Virar",1,1,16,0,6,0,18,0,0,0,0,0,0,0,0,0,0,0,0,6,48,3.9],
    ["Vijayawada",9,0,75,6,0,0,19,0,0,0,0,0,0,0,0,0,0,0,0,72,181,12.1],
    ["Vishakhapatnam",13,15,358,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,40,427,24.7]
], columns=["City","Personal Revenge","Anger","Fraud","Extortion","Causing Disrepute",
           "Prank","Sexual Exploitation","Political Motives","Terrorist Activities",
           "Terrorist Recuit- ment","Terrorist Funding","Inciting Hate against Country",
           "Disrupt Public Service","Sale purchase illegal drugs","Developing own business",
           "Spreading Piracy","Psycho or Pervert","Steal Information","Abetment to Suicide",
           "Others","Total","Crime Rate"])

# Prepare features (X) and target (y)
X = data.drop(['City', 'Total', 'Crime Rate'], axis=1)
y = data['Crime Rate']
feature_names = X.columns.tolist()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Create and train the model with adjusted parameters
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Calculate performance metrics
train_pred = rf_model.predict(X_train)
test_pred = rf_model.predict(X_test)
train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Create visualization
plt.figure(figsize=(20,10))
plot_tree(rf_model.estimators_[0], 
          feature_names=feature_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.savefig('cybercrime_decision_tree.png', dpi=300, bbox_inches='tight')

# Print results
print("\nModel Performance:")
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

print("\nA decision tree visualization has been saved as 'cybercrime_decision_tree.png'")

# Optional: Print actual vs predicted values for test set
test_results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': test_pred
})
print("\nTest Set Predictions:")
print(test_results)