# Step 1: Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 2: Load Dataset
df = pd.read_csv("dataset.csv")

# Step 3: Data Cleaning
# Renaming columns if needed to match a standard format (optional)
# Example: df.rename(columns={'bedrooms': 'Bedrooms'}, inplace=True)

# Drop rows with missing values in relevant columns
features = ['SqFt', 'Bedrooms', 'Bathrooms', 'LivingRoom', 'YearBuilt', 'Garage', 'Zipcode']
target = 'Price'
df = df.dropna(subset=features + [target])

# Step 4: Data Analysis
print("\nBasic Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nChecking for null values:")
print(df.isnull().sum())

# Step 5: Data Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df[['Price', 'SqFt', 'Bedrooms', 'Bathrooms', 'LivingRoom', 'YearBuilt']])
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()

# Step 6: Feature Engineering
# One-hot encode Zipcode (categorical variable)
X = df[features]
X = pd.get_dummies(X, columns=['Zipcode'], drop_first=True)
y = df[target]

# Step 7: Model Selection and Splitting
# Linear Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Model Implementation
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Step 10: Visualization of Predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs. Actual House Prices")
plt.grid(True)
plt.show()