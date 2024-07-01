import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

X= torch.tensor([ [1], [2], [3], [4] ], dtype= torch.float32)
Y= torch.tensor([ [2], [4], [6], [8] ], dtype= torch.float32)
X_test= torch.tensor([5], dtype= torch.float32)
n_sample, n_features= X.shape
input_dim= output_dim= n_features
learning_rate= 0.01
n_iters= 30
model= nn.Linear(input_dim, output_dim)
loss= nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(), lr= learning_rate)
J= []

print(f"Before trainig: f(5)= {model(X_test).item():.3f}")
print(f"Number of samples= {n_sample}\nNumber of features= {n_features}")
[w, b]= model.parameters()
print(f"Initrail: w= {w[0][0].item():.3f}, b= {b[0].item():.3f}\n")

def train():
    for epoch in range(n_iters):
        [w, b]= model.parameters()
        Y_pred= model(X)
        print(f"Epoch {epoch}\ny_pred: {Y_pred.tolist()}= ({w[0][0].item()})*({X.tolist()}) + ({b.item()})")
        
        l= loss(Y, Y_pred)
        J.append(l.item())
        l.backward()
        p= w.item()
        q= b.item()
        optimizer.step()
        print(f"w: {w.item()}= ({p}) - ({learning_rate})*({w.grad.item()})")
        print(f"b: {b.item()}= ({q}) - ({learning_rate})*({b.grad.item()})\n")
        optimizer.zero_grad()

        if epoch%10 == 0:
            [w, b]= model.parameters()
            print(f"w= {w[0][0].item():.3f}, b= {b[0].item():.3f}, loss= {l:.3f}\n")

    print(f"After trainig: f(5)= {model(X_test).item():.3f}")

def plots():
    plt.plot(np.arange(n_iters), J)
    plt.ylabel("J")
    plt.xlabel("iter")
    plt.show()

if __name__=="__main__":
    print("Training Start: ")
    train()
    
    plots()