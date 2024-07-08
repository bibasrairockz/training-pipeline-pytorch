import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

ae_m= pd.read_csv('./data/amazon employee/train.csv')
ae_t= pd.read_csv('./data/amazon employee/test.csv')
ae_m= ae_m.to_numpy(dtype= np.float32)
X_train, Y_train= ae_m[:, 1:], ae_m[:, 0]
X_test, Y_test= ae_m[:, 1:], ae_m[:, 0]
n_samples, n_features= X_train.shape
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
Y_train= Y_train.reshape(Y_train.shape[0], 1)
Y_test= Y_test.reshape(Y_test.shape[0], 1)
X_train= torch.from_numpy(X_train.astype(np.float32))
Y_train= torch.from_numpy(Y_train.astype(np.float32))
X_test= torch.from_numpy(X_test.astype(np.float32))
Y_test= torch.from_numpy(Y_test.astype(np.float32))
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear= nn.Linear(n_input_features, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
model= LogisticRegression(n_features)
learning_rate= 0.05
n_iters= 400
loss= nn.BCELoss()
optimizer= torch.optim.SGD(model.parameters(), lr= learning_rate)
J= []

def train():
    for epoch in range(n_iters):
        Y_pred= model.forward(X_train)

        cost= loss(Y_pred, Y_train)
        cost.backward()
        J.append(cost.item())

        optimizer.step()
        optimizer.zero_grad()

        if (epoch+1)%10 == 0:
            print(f"Epoch {epoch+1}:COST= {cost.item():.5f}")

def accuracy():
    Y_pred= model.forward(X_test)
    Y_pred= Y_pred.round()
    acc= Y_pred.eq(Y_test).sum() / float(Y_test.shape[0])
    print(f"Accuracy= {acc}")

def plot():
    plt.plot(np.arange(n_iters), J)
    plt.xlabel("iter")
    plt.ylabel("J")
    plt.show()

if __name__== "__main__":
    train()
    accuracy()
    plot()
