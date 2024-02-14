from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sqlite3
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

filename = os.path.dirname(__file__) + "/" + 'cleaned_movie_earnings_year_fixed.db'
filename = filename.replace("/", "\\")

conn = sqlite3.connect(filename)
#GET TABLE
query = "SELECT * FROM MOVIE_COMBINED"

# Load and preprocess your movie data (X) and earnings (y)
# X should be a DataFrame with features, and y should be the target variable (earnings)
df = pd.read_sql_query(query, conn)
X = df.select_dtypes(include = 'float64')  
y = X['Worldwide'].values  # Dependent variable
##remove earnings from dataset for training
X = X.drop('Domestic', axis=1)
X = X.drop('Worldwide', axis=1)
X['Year'] = df['Year']    
print(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_model.predict(X_test)

# Evaluate the model using a suitable metric
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Assuming y_test contains the actual earnings and y_pred contains the predicted earnings
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual Earnings vs. Predicted Earnings')
plt.xlabel('Actual Earnings')
plt.ylabel('Predicted Earnings')
plt.grid(True)
plt.show()