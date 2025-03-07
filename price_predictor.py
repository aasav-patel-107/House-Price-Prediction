import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 2: Load dataset
df = pd.read_csv("dataset.csv")

# Step 3: Select relevant features
features = ['sqft_living', 'bedrooms', 'bathrooms', 'year_built', 'zipcode', 'Latitude', 'Longitude']
target = 'price'

# Ensure the selected features exist in the dataset
df = df.dropna(subset=features + [target])  # Remove rows with missing values

X = df[features]
y = df[target]

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Step 8: Visualize results
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs. Actual House Prices")
plt.show()
