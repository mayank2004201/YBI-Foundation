import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
weights = np.array([1200, 1400, 1600, 1800, 2000, 2200, 2400]).reshape(-1, 1)  # Reshaping for sklearn
engine_sizes = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]).reshape(-1, 1)
mileage = np.array([18, 20, 22, 24, 26, 28, 30])
weights_train, weights_test, engine_train, engine_test, mileage_train, mileage_test = train_test_split(
    weights, engine_sizes, mileage, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(np.concatenate((weights_train, engine_train), axis=1), mileage_train)
mileage_pred = model.predict(np.concatenate((weights_test, engine_test), axis=1))
print("Actual Mileage vs Predicted Mileage:")
for actual, predicted in zip(mileage_test, mileage_pred):
    print(f"Actual: {actual} MPG, Predicted: {predicted:.2f} MPG")
try:
    example_weight = float(input("Enter the vehicle weight in kg: "))  # kg
    example_engine_size = float(input("Enter the engine size in liters: "))  # liters
    example_input = np.array([[example_weight, example_engine_size]])
    predicted_mileage = model.predict(example_input)
    print(f"\nPredicted mileage for a vehicle weighing {example_weight} kg with an engine size of {example_engine_size} liters: {predicted_mileage[0]:.2f} MPG")
except ValueError:
    print("Invalid input. Please enter numeric values.")
data = {'Weight (kg)': weights.flatten(), 'Engine Size (liters)': engine_sizes.flatten(), 'Mileage (MPG)': mileage}
df = pd.DataFrame(data)
print("\nData:")
print(df)
