import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
data = pd.read_csv("House Price Prediction Dataset.csv")

print("First 5 rows of the dataset:")
print(data.head())

# Clean Garage column and convert Yes/No into 1/0
data["Garage"] = (
    data["Garage"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({"yes": 1, "no": 0})
)

# Convert Garage column to numeric
data["Garage"] = pd.to_numeric(data["Garage"], errors="coerce")

# Remove rows with missing values
data = data.dropna(
    subset=[
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Floors",
        "YearBuilt",
        "Garage",
        "Price"
    ]
)

# Input features
X = data[
    [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Floors",
        "YearBuilt",
        "Garage"
    ]
]

# Target variable
y = data["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict house prices
y_pred = model.predict(X_test)

# Calculate results
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Results:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Show actual and predicted prices
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices:")
print(results.head(10))

# Plot actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Sample house for prediction
sample_house = pd.DataFrame({
    "Area": [2000],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Floors": [2],
    "YearBuilt": [2015],
    "Garage": [1]
})

# Predict sample house price
predicted_price = model.predict(sample_house)

print("\nPredicted Price for Sample House:", predicted_price[0])