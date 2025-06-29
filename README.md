
# Neural Network Classifiers for Fashion & Digit MNIST Datasets

This repository presents two feedforward neural networks implemented from scratch using purely **NumPy** and **Python** without the aid of any mordern liberaries like PyTorch or TensorFlow. The primary focus is on the Fashion MNIST classifier, followed by the MNIST digit classifier. Each model incorporates modern techniques such as regularization, dropout, mini-batch training, and tailored initialization.

---

## 📈 Summary of Results

| Dataset         | Training Accuracy | Test Accuracy |
| --------------- | ----------------- | ------------- |
| Fashion MNIST   | 90.1%             | 89.0%         |
| MNIST Digits    | 99.9%             | 97.8%         |

---

## 1. Fashion MNIST Classifier

A three-layer network enhanced with Xavier initialization, dropout, and L2 regularization to classify 10 categories of clothing items.

### Architecture

| Layer       | Units | Activation | Initialization         | Regularization        | Dropout |
| ----------- | ----- | ---------- | ---------------------- | --------------------- | ------- |
| **Input**   | 784   | —          | —                      | —                     | —       |
| **Hidden 1**| 300   | ReLU       | Xavier‑Uniform         | L2 (λ = 1e‑4)         | 0.2     |
| **Hidden 2**| 100   | ReLU       | Xavier‑Uniform         | L2 (λ = 1e‑4)         | 0.2     |
| **Output**  | 10    | Softmax    | —                      | —                     | —       |

### Training Configuration
- **Loss Function**: Categorical Cross-Entropy + L2 regularization  
- **Optimizer**: Adam (lr = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 1e‑7, decay = 1e‑5)  
- **Batch Size**: 64  
- **Epochs**: 26  
- **Data Shuffling**: Random permutation each epoch  

---

## 2. MNIST Digit Classifier

A three-layer dense network designed for handwritten digit recognition, incorporating L2 regularization.

### Architecture

| Layer       | Units | Activation | Initialization | Regularization        | Dropout |
| ----------- | ----- | ---------- | -------------- | --------------------- | ------- |
| **Input**   | 784   | —          | —              | —                     | —       |
| **Hidden 1**| 128   | ReLU       | Random (0.01σ) | L2 (λ = 1e‑4)         | —       |
| **Hidden 2**| 64    | ReLU       | Random (0.01σ) | L2 (λ = 1e‑4)         | —       |
| **Output**  | 10    | Softmax    | —              | —                     | —       |

### Training Configuration
- **Loss Function**: Categorical Cross-Entropy + L2 regularization  
- **Optimizer**: Adam (lr = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 1e‑7, decay = 1e‑5)  
- **Batching**: Full-batch gradient descent  
- **Epochs**: 1,001  

---

## 📂 Directory Structure
```
├── Digit MNIST/
│   ├── digit_sample.png                  # Sample image for testing
│   ├── DigitMNIST_inference.py           # Inference script for digit prediction
│   ├── DigitMNIST_model_params.pkl       # Trained model parameters for digits
│   └── DigitMNIST_model.py               # Training implementation for MNIST digits
├── Fashion MNIST/
│   ├── fashion_sample.png                # Sample image for testing
│   ├── FashionMNIST_inference.py         # Inference script for digit prediction
│   ├── FashionMNIST_model_params.pkl     # Trained model parameters for digits
│   └── FashionMNIST_model.py             # Training implementation for MNIST digits
├── LICENSE                   # Project license
└── README.md                 # Project documentation

```

---

## 🛠️ Dependencies
- **Python 3.x**  
- **NumPy**  
- **Matplotlib** (for plots)  
- **Pillow** (for image processing)
- **TensorFlow** (for loading MNIST datasets)

---

## 👨‍💻 How to run it on your own computer

1. **Clone Repository**  
   ```bash
   git clone https://github.com/zeeshanhxider/Vanilla-Python-Classifier.git
   cd Vanilla-Python-Classifier
   ```

2. **Install Dependencies**  
   ```bash
   pip install numpy matplotlib pillow tensorflow
   ```

3. **Train the Model (Not necessary since pre-trained model already included)**  
   - **Digits:**  
     ```bash
     cd "Digit MNIST"
     python DigitMNIST_model.py
     ```
   - **Fashion:**  
     ```bash
     cd "Fashion MNIST"
     python FashionMNIST_model.py
     ```

4. **Load Testing Images**  
   - **Digits:**  
   Save your test image as "digit_sample.png" in the MNIST digit directory.
   - **Fashion:**  
   Save your test image as "fashion_sample.png" in the MNIST fashion directory.

**⚠️ Note**: *These models were trained exclusively on the original MNIST and Fashion MNIST datasets. For best results, use test images from these datasets. Custom images may lead to reduced accuracy—especially for Fashion MNIST—unless they closely match the dataset in size, format (28×28 grayscale), and content.*

5. **Run Inference**  
   ```bash
   # For Digit Prediction
   python DigitMNIST_model.py

   # For Fashion Prediction
   python FashionMNIST_model.py 
   ```

---

## Made with lots of complicated maths and 🧡 by zeeshan