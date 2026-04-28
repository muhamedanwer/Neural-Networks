import tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Number of classes (0–9 → 10)
number_of_classes = y_train.max() + 1   # or simply 10

# Scale images to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Add channel dimension: (28, 28) → (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)
input_shape = (28, 28, 1)

# Build the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(number_of_classes, activation="softmax")
])

# Compile
model.compile(
    loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(),
    optimizer="adam",
    metrics=["accuracy"]
)

# Train
model.fit(x_train, y_train, epochs=15, batch_size=128)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('\nTest accuracy: %.2f%%' % (test_acc * 100))