import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sqlite3
import pandas as pd
import os

filename = os.path.dirname(__file__) + "/" + 'cleaned_movie_earnings_year_fixed.db'
filename = filename.replace("/", "\\")

conn = sqlite3.connect(filename)

#GET TABLE
query = "SELECT * FROM MOVIE_COMBINED"
df = pd.read_sql_query(query, conn)
print(df)

# Sample data (you should replace this with your own dataset)
X = df['Budget'].values.reshape(-1, 1).astype(float)  # Independent variable
y = df['Worldwide'].values  # Dependent variable

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)
print(X)
# Make predictions
X_pred = np.array([16_000_000, 380_000_000, 200_000_000]).reshape(-1, 1)  # New data for prediction
y_pred = model.predict(X_pred)

# Visualize the data and regression line
plt.scatter(X, y, label="Data Points")
plt.plot(X_pred, y_pred, color='red', label="Regression Line")
plt.xlabel("Budget")
plt.ylabel("Worldwide Earnings")
plt.legend()
plt.title("Movie Budget vs. Worldwide Earnings")
plt.show()

# Get the coefficients and intercept of the linear regression model
slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope (Coefficient): {slope}")
print(f"Intercept: {intercept}")