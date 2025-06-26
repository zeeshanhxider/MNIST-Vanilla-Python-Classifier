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
with open('MNISTfashion.pkl', 'rb') as f:
    model_data = pickle.load(f)

dense1 = Layer_Dense(784, 128)
dense1.weights = model_data['dense1_weights']
dense1.biases = model_data['dense1_biases']

activation1 = Activation_ReLU()

dense2 = Layer_Dense(128, 10)
dense2.weights = model_data['dense2_weights']
dense2.biases = model_data['dense2_biases']

activation2 = Activation_Softmax()

# Fashion MNIST label names
label_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

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

    probs = activation2.output[0]
    predicted_class = np.argmax(probs)
    confidence = probs[predicted_class] * 100

    print(f"Predicted Class: {label_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nProbabilities:")
    for i, prob in enumerate(probs):
        print(f"  {label_names[i]:<12}: {prob * 100:.2f}%")

#Paste your image path here
image_path = 'item.png'
predict(image_path)