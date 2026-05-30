import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --------------------------------------------------
# Load Boston Dataset
# --------------------------------------------------
boston = fetch_openml(name='boston', version=1, as_frame=True)

df = boston.frame

# Select two features
X = df[['RM', 'AGE']].astype(float).values
y = df['MEDV'].astype(float).values

# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# Multi-Layer Perceptron
# --------------------------------------------------
model = Sequential([
    Dense(16, activation='relu', input_shape=(2,)),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train model
history = model.fit(
    X_scaled,
    y,
    epochs=300,
    batch_size=16,
    verbose=0
)

# --------------------------------------------------
# Predictions
# --------------------------------------------------
y_pred = model.predict(X_scaled, verbose=0)

# --------------------------------------------------
# Generate Surface
# --------------------------------------------------
rm_range = np.linspace(X[:,0].min(), X[:,0].max(), 50)
age_range = np.linspace(X[:,1].min(), X[:,1].max(), 50)

RM_grid, AGE_grid = np.meshgrid(rm_range, age_range)

grid_points = np.c_[RM_grid.ravel(), AGE_grid.ravel()]
grid_scaled = scaler.transform(grid_points)

Z = model.predict(grid_scaled, verbose=0)
Z = Z.reshape(RM_grid.shape)

# --------------------------------------------------
# 3D Visualization
# --------------------------------------------------
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Real data points
ax.scatter(
    X[:,0],
    X[:,1],
    y,
    color='blue',
    alpha=0.6,
    label='Actual Data'
)

# Predicted surface
ax.plot_surface(
    RM_grid,
    AGE_grid,
    Z,
    alpha=0.5
)

ax.set_xlabel('RM')
ax.set_ylabel('AGE')
ax.set_zlabel('MEDV')
ax.set_title('MLP Regression on Boston Housing Dataset')

plt.legend()
plt.show()