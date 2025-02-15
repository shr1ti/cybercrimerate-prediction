# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Data for classification models
classifiers = ['Logistic Regression', 'Naive Bayes', 'Decision Tree']
accuracy = [85.71, 86, 86]
precision = [66.67, 86, 86]
recall = [98.29, 100, 100]
f1_score = [80.00, 92, 92]

# Data for regression models
regressors = ['SVM', 'KNN (k=4)', 'Random Forest']
mse = [23.52, 16.7907, 0]  # Replace None with 0 or remove it
r2_score = [0.0725, 0.3380, 0]  # Replace None with 0 or remove it
training_rmse_rf = 1.56
testing_rmse_rf = 0.75

# Plot classification metrics
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(classifiers))  # the label locations
width = 0.2  # the width of the bars

# Plot Accuracy, Precision, Recall, and F1-Score for classifiers
ax[0].bar(x - width * 1.5, accuracy, width, label='Accuracy', color='skyblue')
ax[0].bar(x - width / 2, precision, width, label='Precision', color='salmon')
ax[0].bar(x + width / 2, recall, width, label='Recall', color='limegreen')
ax[0].bar(x + width * 1.5, f1_score, width, label='F1-Score', color='mediumpurple')

# Label classification plot
ax[0].set_xlabel('Classification Models')
ax[0].set_ylabel('Percentage (%)')
ax[0].set_title('Classification Model Performance Comparison')
ax[0].set_xticks(x)
ax[0].set_xticklabels(classifiers)
ax[0].legend(loc='upper right')

# Correct regression model plotting
x = np.arange(len(regressors))  # Ensure all regressors are included
ax[1].bar(x - width / 2, mse, width, label='Mean Squared Error (MSE)', color='gold')
ax[1].bar(x + width / 2, r2_score, width, label='R² Score', color='coral')

# Adding Random Forest RMSE to the plot
ax[1].bar(x[-1] - width / 2, training_rmse_rf, width, label='Training RMSE (Random Forest)', color='lightblue')
ax[1].bar(x[-1] + width / 2, testing_rmse_rf, width, label='Testing RMSE (Random Forest)', color='steelblue')

# Label regression plot
ax[1].set_xlabel('Regression Models')
ax[1].set_ylabel('Score')
ax[1].set_title('Regression Model Performance Comparison')
ax[1].set_xticks(x)
ax[1].set_xticklabels(regressors)
ax[1].legend(loc='upper right')

plt.tight_layout()
plt.show()
