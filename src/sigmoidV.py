import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate values for z (input to the sigmoid function)
z = np.linspace(-10, 10, 100)

# Apply the sigmoid function to z
sigmoid_values = sigmoid(z)

# Plot the sigmoid curve
plt.figure(figsize=(8, 6))
plt.plot(z, sigmoid_values, color='blue')
plt.title('Sigmoid Function for Logistic Regression')
plt.xlabel('Input (z)')
plt.ylabel('Output (Probability)')
plt.grid(True)

# Add horizontal and vertical lines at the threshold of 0.5 and input 0
plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold = 0.5')
plt.axvline(x=0, color='green', linestyle='--', label='z = 0')

plt.legend()
plt.show()
