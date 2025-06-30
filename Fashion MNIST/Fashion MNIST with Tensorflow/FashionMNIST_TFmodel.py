import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#load the fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#normalize and flatten the images
X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28*28).astype('float32') / 255.0

#convert labels to categorical one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

#build the model
model = models.Sequential([
    layers.Dense(300, activation='relu',  kernel_regularizer=regularizers.l2(1e-4), input_shape=(28*28,)),
    layers.Dropout(0.2),  # Add dropout layer for regularization
    layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.2),  # Add another dropout layer
    layers.Dense(10, activation='softmax')
])

#compile the model
model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#train the model
history = model.fit(
    X_train, y_train,
    epochs=24,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=2
)

#save the model
model.save('FashionMNIST_model_params.keras')

# Plot training metrics
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='orange')
plt.plot(history.history['val_loss'], label='Test Loss', color='red')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()