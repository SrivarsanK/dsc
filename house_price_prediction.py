import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the dataset
csv_file = "house_cleaned.csv"
if not os.path.exists(csv_file):
    print(f"Dataset file {csv_file} not found.")
    exit()

# Read the dataset
df = pd.read_csv(csv_file)

# Display initial price statistics
print("\nInitial price statistics (in Crores):")
print(df['price'].describe())

# Clean and convert the balcony column
df['balcony'] = df['balcony'].replace('3+', '3')
df['balcony'] = pd.to_numeric(df['balcony'], errors='coerce')

# Select and create relevant features
numeric_features = ['price_per_sqft', 'area', 'bedRoom', 'bathroom', 'balcony']
target = 'price'

# Create feature dataframe
X = df[numeric_features]
y = df[target]

# Remove any rows with missing values
df_clean = pd.concat([X, y], axis=1).dropna()
X = df_clean[numeric_features]
y = df_clean[target]

# Log transform numeric features to handle skewness
X_logged = X.copy()
X_logged['price_per_sqft'] = np.log1p(X['price_per_sqft'])
X_logged['area'] = np.log1p(X['area'])
y_logged = np.log1p(y)

# Print dataset info
print("\nDataset shape:", df_clean.shape)
print("\nFeature columns info:")
print(X_logged.info())
print("\nSample of preprocessed data:")
print(X_logged.head())

# Scale features
scaler_X = RobustScaler()
X_scaled = scaler_X.fit_transform(X_logged)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_logged, test_size=0.2, random_state=42
)

# Create and compile model with adjusted architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(1)
])

# Compile with adjusted learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Add early stopping with increased patience
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Make predictions
y_pred = model.predict(X_test).flatten()

# Convert predictions back to original scale
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred)

# Calculate metrics on original scale
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_original)

print("\nModel Evaluation (Original Scale):")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared score: {r2:.4f}")

# Print sample predictions
print("\nSample Predictions (in Crores):")
sample_indices = np.random.randint(0, len(y_test), 5)
for idx in sample_indices:
    actual = y_test_original.iloc[idx]
    predicted = y_pred_original[idx]
    print(f"Actual: {actual:.2f} Cr, Predicted: {predicted:.2f} Cr")

# Plot training history
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# MAE plot
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()
