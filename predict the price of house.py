import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample data
data = {
    'SquareFootage': [1500, 2000, 2500, 1800, 2200, 2700],
    'Bedrooms': [3, 4, 3, 2, 4, 5],
    'Bathrooms': [2, 3, 2, 2, 3, 4],
    'Price': [300000, 450000, 400000, 350000, 500000, 600000]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Displaying the results
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Example prediction
example_house = [[2500, 3, 2]]
predicted_price = model.predict(example_house)
print(f"Predicted price for house with 2500 sq ft, 3 bedrooms, and 2 bathrooms: ${predicted_price[0]:,.2f}")
