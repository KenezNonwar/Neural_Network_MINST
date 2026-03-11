# Neural Network MNIST Digit Classifier

A simple **Neural Network implementation for handwritten digit recognition** using the **MNIST dataset**.

This project demonstrates how to build, train, and evaluate a neural network for image classification using Python and deep learning libraries.

The model learns to classify grayscale images of handwritten digits from **0–9**.

---

# Project Overview

This project covers the complete pipeline of a basic deep learning model:

1. Load and preprocess the MNIST dataset
2. Normalize pixel values
3. Build a neural network model
4. Train the model
5. Evaluate performance
6. Predict digits from test data

The goal of this project is to **understand the fundamentals of neural networks and image classification**.

---

# Dataset

The **MNIST dataset** consists of:
  LINK: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

- 60,000 training images  
- 10,000 test images  
- 28×28 grayscale images  
- 10 classes (digits 0–9)

Each image is flattened into a **784-dimension vector (28×28)** before being fed into the neural network.

---

# Model Architecture

```
Input Layer: 784 neurons (28x28 pixels)

Hidden Layer:
Dense Layer (ReLU activation)

Output Layer:
Dense Layer (10 neurons, Softmax activation)
```

The **Softmax function** converts the outputs into probabilities for each digit class.

---

# Technologies Used

- Python
- NumPy
- Matplotlib

---

# Installation

Clone the repository:

```bash
git clone https://github.com/KenezNonwar/Neural_Network_MINST.git
cd Neural_Network_MINST
```

Install dependencies:

```bash
pip install numpy matplotlib pandas
```

---

# Running the Project

Run the Python script:

```bash
python main.py
```


Steps performed by the script:

1. Load MNIST dataset
2. Preprocess images
3. Build neural network
4. Train the model
5. Evaluate accuracy
6. Predict digits

---

# Example Output

```
Epoch 1/10
loss: 0.32
accuracy: 0.91

Epoch 10/10
loss: 0.05
accuracy: 0.98
```

Typical accuracy for a simple neural network on MNIST:

**~97–98% accuracy**

---

# Learning Goals

This project helps understand:

- Neural network fundamentals
- Data preprocessing for images
- Model training and evaluation
- Classification tasks
- Deep learning workflow

---

# Future Improvements

Possible improvements:

- Implement a **Convolutional Neural Network (CNN)**
- Add **data visualization**
- Create a **digit drawing interface**
- Hyperparameter tuning
- Model deployment

---

# Author

**Ken Nonwar**  
AI / Machine Learning 

GitHub:  
https://github.com/KenezNonwar
