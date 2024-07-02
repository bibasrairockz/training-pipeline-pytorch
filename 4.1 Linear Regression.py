import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

X_numpy, Y_numpy= datasets.make_regression(n_samples= 100, n_features= 1 , noise= 20, random_state= 1)
X= torch.from_numpy(X_numpy.astype(np.float32))
Y= torch.from_numpy(Y_numpy.astype(np.float32))
n_samples, n_fleatures= X.shape
input_dim= output_dim= n_fleatures
Y= Y.view(Y.shape[0], 1)
model= nn.Linear(input_dim, output_dim)
loss= nn.MSELoss()
learning_rate= 0.01
n_iters= 200
optimizer= torch.optim.SGD(model.parameters(), lr= learning_rate)
J= []

print(f"Initail values: X[0]= {X[0]}, Y[0]= {Y[0]}\n")

def train():
    for epoch in range(n_iters):
        Y_pred= model(X)
        [w, b]= model.parameters()
        print(f"Epoch {epoch}\nY_pred: {Y_pred[0].item()} = ({w[0][0].item()})*({X[0].item()}) + ({b.item()})")
        
        l= loss(Y, Y_pred)
        J.append(l.item())
        l.backward()

        p= w[0][0].item()
        q= b[0].item()
        optimizer.step()
        print(f"w_new: {w[0][0].item()} = {p} - ({learning_rate})*({w.grad.item()})")
        print(f"b_new: {b[0].item()} = {q} - ({learning_rate})*({b.grad.item()})") 
        optimizer.zero_grad()

       
        if epoch%10== 0:
            [w, b]= model.parameters()
            print(f"New: w= {w[0][0]}, b= {b[0]}, loss= {l}")

def plots():
    Y_pred= model(X)
    plt.plot(X.detach().numpy(), Y.detach().numpy(), "ro")
    plt.plot(X.detach().numpy(), Y_pred.detach().numpy(), "b")
    plt.show()

    plt.plot(np.arange(n_iters), J)
    plt.show()


if __name__=="__main__":
    train()

    plots()