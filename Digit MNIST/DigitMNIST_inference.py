import numpy as np
import pickle
from PIL import Image

# coding only the forward pass of the neural network
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.zeros((n_inputs, n_neurons))  
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Load the pre-trained model
with open('DigitMNIST_model_params.pkl', 'rb') as f:
    model_data = pickle.load(f)

dense1 = Layer_Dense(784, 128)
dense1.weights = model_data['dense1_weights']
dense1.biases = model_data['dense1_biases']

activation1 = Activation_ReLU()

dense2 = Layer_Dense(128, 64)
dense2.weights = model_data['dense2_weights']
dense2.biases = model_data['dense2_biases']

activation2 = Activation_ReLU()

dense3 = Layer_Dense(64, 10)
dense3.weights = model_data['dense3_weights']
dense3.biases = model_data['dense3_biases']

activation3 = Activation_Softmax()

def load_and_prepare_image(path):
    img = Image.open(path).convert('L')           # Convert to grayscale
    img = img.resize((28, 28))                    # Resize to 28x28
    img_array = np.array(img).astype('float32')   # Convert to numpy array

    avg_pixel = np.mean(img_array)
    if avg_pixel > 127:                           # If background is light 
        img_array = 255 - img_array               # Invert
        img = Image.fromarray(255 - np.array(img))

    img_array /= 255.0                            # Normalize to 0–1
    img_array = img_array.reshape(1, 784)         # Flatten
    img.show()
    return img_array

def predict(path):
    x = load_and_prepare_image(path)
    
    # Forward pass
    dense1.forward(x)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    probabilities = activation3.output[0]
    prediction = np.argmax(probabilities)
    confidence = probabilities[prediction] * 100

    print(f"Predicted digit: {prediction}")
    print(f"Confidence: {confidence:.2f}%")

    # Optional: show all class probabilities
    print("\nClass probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"{i}: {prob*100:.2f}%")

# paste image file path here:
image_path = 'digit_sample.png'
predict(image_path)