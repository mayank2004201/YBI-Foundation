# Importing necessary libraries
import numpy as np
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

# Printing the results
print("Actual Mileage vs Predicted Mileage:")
for actual, predicted in zip(mileage_test, mileage_pred):
    print(f"Actual: {actual} MPG, Predicted: {predicted:.2f} MPG")

# Example prediction
example_weight = np.array([[2000]])  # kg
example_engine_size = np.array([[2.0]])  # liters
predicted_mileage = model.predict(np.concatenate((example_weight, example_engine_size), axis=1))
print(f"\nPredicted mileage for a vehicle weighing {example_weight[0][0]} kg with an engine size of {example_engine_size[0][0]} liters: {predicted_mileage[0]:.2f} MPG")
