Overview
This project demonstrates a simple linear regression model for predicting house prices based on features such as square footage, number of bedrooms, and number of bathrooms. The model is implemented using Python and the scikit-learn library. The goal is to predict house prices given the features, and evaluate the model's performance.

Dependencies
To run this project, you'll need to install the following Python packages:

pandas
scikit-learn
You can install the dependencies using pip:

bash
Copy code
pip install pandas scikit-learn
Dataset
The dataset used in this project is a small sample dataset with the following columns:

SquareFootage: The area of the house in square feet.
Bedrooms: The number of bedrooms in the house.
Bathrooms: The number of bathrooms in the house.
Price: The price of the house in USD.
Sample Data
python
Copy code
data = {
    'SquareFootage': [1500, 2000, 2500, 1800, 2200, 2700],
    'Bedrooms': [3, 4, 3, 2, 4, 5],
    'Bathrooms': [2, 3, 2, 2, 3, 4],
    'Price': [300000, 450000, 400000, 350000, 500000, 600000]
}
Workflow
Data Preparation:

The data is loaded into a Pandas DataFrame.
The features (SquareFootage, Bedrooms, Bathrooms) and the target variable (Price) are separated.
Train-Test Split:

The dataset is split into training (80%) and testing (20%) sets using train_test_split.
Model Training:

A LinearRegression model is created and trained on the training data.
Prediction and Evaluation:

The model predicts house prices on the test set.
The performance of the model is evaluated using Mean Squared Error (MSE) and R^2 Score.
Example Prediction:

The model predicts the price for a house with 2500 square feet, 3 bedrooms, and 2 bathrooms.
Results
After training the model, the following metrics are displayed:

Mean Squared Error (MSE): This measures the average squared difference between the actual and predicted values. A lower value indicates better performance.

R^2 Score: This represents the proportion of variance in the dependent variable that is predictable from the independent variables. A value closer to 1 indicates a better fit.

Example output:

yaml
Copy code
Mean Squared Error: 312500000.00
R^2 Score: 0.84
Predicted price for house with 2500 sq ft, 3 bedrooms, and 2 bathrooms: $425,000.00
Usage
You can modify the example house features to predict prices for different house configurations by changing the example_house variable:

python
Copy code
example_house = [[2500, 3, 2]]
predicted_price = model.predict(example_house)
This will output the predicted price for a house with the given specifications.

Conclusion
This project provides a basic demonstration of linear regression for predicting house prices. While the dataset is small, the model can be scaled and applied to larger datasets for more accurate predictions.
