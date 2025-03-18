import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Download the dataset from Kaggle using kagglehub
try:
    path = kagglehub.dataset_download("jenilhareshbhaighori/house-price-prediction")
    print("Path to dataset files:", path)
except Exception as e:
    print("Error downloading dataset from kagglehub:", e)

# Step 2: Load the dataset from the downloaded path
# The path returned by kagglehub.dataset_download might contain a zipped file which needs to be unzipped.
# However, the documentation for kagglehub might provide more details on what the returned path contains.
# For now, let us assume that the path points to a directory where the dataset is stored.
# However, the actual file name might need to be checked manually if the kagglehub package does not provide a specific file name.

# For now, let us assume that the actual CSV file is named "House_Price_India.csv" based on the initial part of the question.
# However, if the "jenilhareshbhaighori/house-price-prediction" dataset has a different file name, it should be used instead.
# However, the question mentioned that the "India Housing dataset" should be used so it seems that "House_Price_India.csv" should be the main file.
df = pd.read_csv(path + "/House_Price_India.csv")

# Step 3: Visualize the data for better understanding (basic visualizations)
# Visualize the distribution of the target variable (Price)
sns.histplot(df['Price'], kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.show()

# Visualize some of the features against the target variable (e.g., number of bedrooms, square footage, etc.)
sns.scatterplot(x='number of bedrooms', y='Price', data=df)
plt.title("Number of Bedrooms vs. Price")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Price")
plt.show()

sns.scatterplot(x='living area', y='Price', data=df)
plt.title("Living Area vs. Price")
plt.xlabel("Living Area")
plt.ylabel("Price")
plt.show()

# Step 4: Preprocess the data
# Drop unnecessary columns such as the "id" column if it exists
df = df.drop(columns=['id'])

# Separate features and target variable
X = df.drop(columns=['Price']).values
y = df['Price'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Design a neural network with at least one hidden layer
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

# Step 6: Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 7: Train the model for a suitable number of epochs
model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)

# Step 8: Evaluate the model's performance
y_pred = model.predict(X_test).flatten()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared score: {r2}")