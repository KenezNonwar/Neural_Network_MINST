import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH='mnist_train.csv'
def start():
    data=pd.read_csv(PATH)
    data=np.array(data)
    global m,n
    m,n=data.shape
    # print(m,n)
    np.random.shuffle(data)
    train_data=data[0 : int(0.8*m), : ]
    val_data=data[int(0.8*m):m, : ]

    X_train=train_data[ : , 1: ].T #A[] matrix
    X_train=X_train/255.0 #normalizing 0-1
    y_train=train_data[ : , 0 ]

    X_val=val_data[:,1:].T
    X_val=X_val/255.0
    y_val=val_data[ : , 0 ]
    return X_train,y_train,X_val,y_val

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_val.shape)
    # print(y_val.shape)

def ReLU(X):
    return np.maximum(X,0)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def one_hot_encoder(y):
    one_hot_y=np.zeros((y.size,y.max()+1))
    one_hot_y[np.arange(y.size),y]=1
    return one_hot_y.T

def init_parameter():
    W1=np.random.rand(10,784)-0.5
    B1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    B2 = np.random.rand(10, 1) - 0.5
    return W1, B1, W2, B2

def forward_propagation(W1, B1, W2, B2, X):
    Z1=W1.dot(X)+B1
    A1=ReLU(Z1)
    Z2=W2.dot(A1)+B2
    A2=softmax(Z2)
    return Z1, A1, Z2, A2

def backward_propagation(Z1, Z2, A1, A2, W2, X, Y):
    m = X.shape[1]
    dZ2=A2-one_hot_encoder(Y)
    dW2=1 / m * dZ2.dot(A1.T)
    dB2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = (W2.T).dot(dZ2) * (Z1 > 0)
    dW1=1/m* dZ1.dot(X.T)
    dB1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, dB1, dW2, dB2

def update(W1,B1,W2,B2,dW1, dB1, dW2, dB2,learn_rate):
    W1 = W1 - learn_rate * dW1
    B1 = B1 - learn_rate * dB1
    W2 = W2 - learn_rate * dW2
    B2 = B2 - learn_rate * dB2
    return W1, B1, W2, B2

def predict(A2):
    return np.argmax(A2,0)

def accuracy(pred,y):
    return np.sum(pred==y)/y.size

def grad_decent(X,y,alpha,n_iter):
    W1, B1, W2, B2= init_parameter()
    for i in range(n_iter):
        Z1, A1, Z2, A2 = forward_propagation(W1, B1, W2, B2, X)
        dW1, dB1, dW2, dB2 = backward_propagation(Z1, Z2, A1, A2, W2, X, y)
        W1, B1, W2, B2 = update(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha)
        if i%20==0:
            print('iteration no:',i)
            print('Accuracy:',accuracy( predict( A2 ) , y ))

    return W1, B1, W2, B2

def save_model(W1, B1, W2, B2):
    np.savez("mnist_model.npz", W1=W1, B1=B1, W2=W2, B2=B2)


def main():
    X_train,y_train,X_val,y_val = start()
    W1, B1, W2, B2 =grad_decent( X_train, y_train,0.1, 1000)
    save_model(W1, B1, W2, B2)

if __name__=='__main__':
    main()