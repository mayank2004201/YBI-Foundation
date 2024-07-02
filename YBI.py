# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample data: vehicle weight in kg and engine size in liters
weights = np.array([1200, 1400, 1600, 1800, 2000, 2200, 2400]).reshape(-1, 1)  # Reshaping for sklearn
engine_sizes = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]).reshape(-1, 1)
mileage = np.array([18, 20, 22, 24, 26, 28, 30])

# Splitting data into training and testing sets
weights_train, weights_test, engine_train, engine_test, mileage_train, mileage_test = train_test_split(
    weights, engine_sizes, mileage, test_size=0.2, random_state=0)

# Creating a linear regression model
model = LinearRegression()

# Training the model
model.fit(np.concatenate((weights_train, engine_train), axis=1), mileage_train)

# Making predictions
mileage_pred = model.predict(np.concatenate((weights_test, engine_test), axis=1))

# Plotting the results
plt.figure(figsize=(10, 6))

plt.scatter(weights_test, mileage_test, color='blue', label='Actual Mileage')
plt.scatter(engine_test, mileage_test, color='green', label='Actual Mileage')
plt.scatter(weights_test, mileage_pred, color='red', label='Predicted Mileage')

plt.title('Mileage Prediction')
plt.xlabel('Vehicle Weight (kg) and Engine Size (liters)')
plt.ylabel('Miles per gallon (MPG)')
plt.legend()
plt.show()

# Example prediction
example_weight = np.array([[2000]])  # kg
example_engine_size = np.array([[2.0]])  # liters
predicted_mileage = model.predict(np.concatenate((example_weight, example_engine_size), axis=1))
print(f"Predicted mileage for a vehicle weighing {example_weight[0][0]} kg with an engine size of {example_engine_size[0][0]} liters: {predicted_mileage[0]} MPG")
