import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt  

weight=np.array([1200,1400,1600,1800,2000,2200,2400]).reshape(-1,1)
engine=np.array([1.0,1.2,1.4,1.6,1.8,2.0,2.2]).reshape(-1,1)
mileage=np.array([18,20,22,24,26,28,30])
weight_train,weight_test,engine_train,engine_test,mileage_train,mileage_test=train_test_split(
    weight,engine,mileage,test_size=0.2,random_state=0)
#creation of linear regression model 
model = LinearRegression()
model.fit(np.concatenate((weight_train,engine_train), axis=1), mileage_train)
mileage_predict=model.predict(np.concatenate((weight_test,engine_test), axis=1))
print("Actual Mileage vs Predicted Mileage:")
for actual, predicted in zip(mileage_test, mileage_predict):
    print(f"Actual: {actual} MPG , Predicted: {predicted:.2f} MPG")
#code fot taking the input from the user and running it 
try:
    example_weight = float(input("Enter the vehicle weight(in kg): "))
    example_engine_size = float(input("Enter the vehicle engine size(in liters): "))
    example_input = np.array([[example_weight,example_engine_size]])
    predicted_mileage = model.predict(example_input)
    # 'f' is a type of sring which allows to add expressions and literals in a single string 
    # in f string the expressions are evaluated at run time and are then added to the string 
    print(f"\nPredicted mileage of the vehicle is = {predicted_mileage[0]:.2f} MPG")
except ValueError:
    print("Invalid input, check the input values")

#  Plotting the data and user input
plt.figure(figsize=(10, 6))
plt.scatter(weight.flatten(), mileage, color='blue', label='Actual Data', marker='o')
plt.scatter(weight_test.flatten(), mileage_predict, color='orange', label='Predicted Mileage', marker='x')
# Plotting the graph with the respect to the user's input
plt.scatter(example_weight, predicted_mileage, color='red', label='Input Prediction', marker='D', s=100)
# labelling the graph
plt.title('Vehicle Mileage Prediction System')
plt.xlabel('Vehicle Weight (kg)')
plt.ylabel('Mileage (MPG)')
plt.legend()
plt.grid()
plt.show()
data={'weight': weight.flatten(),'engine':engine.flatten(),'mileage': mileage}
df=pd.DataFrame(data)
print("\nData:")
print(df)    
