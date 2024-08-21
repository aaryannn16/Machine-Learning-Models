import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Dataset used
data = pd.read_csv('Car_Data.csv')

# Creating a DataFrame from the Dataset
df = pd.DataFrame.dropna(data)

# Defining features (X) and target (y)
X = df[['Power(bhp)', 'Engine(CC)','Seats','Year','Mileage','Kilometers_Driven']]
y = df['Price(lakhs)']

# Splitting the data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating and training the model for Prediction
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction done on Test Set
y_pred = model.predict(X_test)

# Calculating metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Predicting Values with user input
def Predict_Model(model):
    print("\nEnter values for new prediction:")
    feature1 = float(input("Enter value for Power(bhp): "))
    feature2 = float(input("Enter value for Engine(CC): "))
    feature3 = float(input("Enter value for Seats: "))
    feature4 = float(input("Enter value for Year: "))
    feature5 = float(input("Enter value for Mileage: "))
    feature6 = float(input("Enter value for Kilometers Driven: "))
    
    new_data = np.array([[feature1, feature2,feature3,feature4,feature5,feature6]])
    prediction = model.predict(new_data)
    
    print(f"Predicted Target value for Price (in Lakhs): Rs.{prediction[0]}")

Predict_Model(model)

# Plotting the Actual vs Predicted values on graph
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='red',marker = '+')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()