
# MNIST & Fashion MNIST Neural Network from Scratch 

This repository contains a fully implemented feedforward neural network built from scratch in Python using only NumPy. The network is trained and tested on the MNIST digits dataset and the Fashion MNIST dataset. No high-level deep learning libraries are used for the core training loops or architecture.

## 🧠 Architecture & Design

| Layer            | Units | Activation          |
| ---------------- | ----- | ------------------- |
| Input (Flatten)  | 784   | —                   |
| Hidden           | 128   | ReLU                |
| Output           | 10    | Softmax             |

- **Loss:** Categorical Cross-Entropy  
- **Optimizer:** Adam  
- **Epochs:** 1001  
- **Metrics:** Training Loss, Training & Test Accuracy  

---

## 📈 Results

- **MNIST Digits**  
  - Training Accuracy: **97.4%**  
  - Test Accuracy: **99.6%**  

- **Fashion MNIST**  
  - Training Accuracy: **91.1%**  
  - Test Accuracy: **87.4%**  

---

## Directory Structure
```
├── digit/
│   ├── train_digit.py        # Training implementation for MNIST digits
│   ├── infer_digit.py        # Inference script for digit prediction
│   ├── MNISTdigit.pkl        # Trained model parameters for digits
│   └── digit.png             # Sample input image for inference
├── fashion/
│   ├── train_fashion.py      # Training implementation for Fashion MNIST
│   ├── infer_fashion.py      # Inference script for fashion item prediction
│   ├── FashionMNIST.pkl      # Trained model parameters for fashion
│   └── fashion.png           # Sample input image for inference
└── README.md                 # Project documentation
```

---

## 🚀 How to run it on your own computer

1. **Clone Repository**  
   ```bash
   git clone https://github.com/zeeshanhxider/Vanilla-Python-Classifier.git
   cd Vanilla-Python-Classifier
   ```

2. **Install Dependencies**  
   ```bash
   pip install numpy matplotlib pillow tensorflow
   ```

3. **Train the Model**  
   - **Digits:**  
     ```bash
     cd MNIST digit
     python MNISTdigit.py
     ```
   - **Fashion:**  
     ```bash
     cd MNIST fashion
     python MNISTfashion.py
     ```

4. **Load Testing Images**  
   - **Digits:**  
   Save your test image as "digit.png" in the MNIST digit directory.
   - **Fashion:**  
   Save your test image as "item.png" in the MNIST fashion directory.

5. **Run Inference**  
   ```bash
   # For Digit Prediction
   python MNISTdigittest.py 

   # For Fashion Prediction
   python MNISTfashiontest.py 
   ```

---


## 🛠️ Dependencies
- **Python 3.x**  
- **NumPy**  
- **Matplotlib** (for plots)  
- **Pillow** (for image processing)

---

**Made with lots of complicated maths and 🧡 by zeeshan**
