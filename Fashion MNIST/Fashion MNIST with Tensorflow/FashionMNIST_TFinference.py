import numpy as np
import tensorflow as tf
from PIL import Image

#fashion mnist class names
label_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load the pre-trained model
model = tf.keras.models.load_model('FashionMNIST_model_params.keras')

#preprocess the image
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

#predict and display the result
def predict(path):
    x = load_and_prepare_image(path)
    
    #forward pass
    predictions = model.predict(x)
    prob = predictions[0]
    predicted_class = np.argmax(prob)
    confidence = prob[predicted_class] * 100

    print(f"Predicted class: {label_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nClass probabilities:")
    for i, p in enumerate(prob):
        print(f"{label_names[i]}: {p * 100:.2f}%")

# image path
image_path = 'fashion_sample.png'
predict(image_path)