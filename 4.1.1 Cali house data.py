from sklearn import datasets
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

c_h= datasets.fetch_california_housing()
X_numpy, Y_numpy= c_h.data, c_h.target
n_samples, n_features= X_numpy.shape
sc= StandardScaler()
X_numpy= sc.fit_transform(X_numpy)
Y_numpy= Y_numpy.reshape(Y_numpy.shape[0], 1)
X, Y= torch.from_numpy(X_numpy.astype(np.float32)), torch.from_numpy(Y_numpy.astype(np.float32))
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.15, random_state= 1)
print(X_train[0], Y_train[0])
class LinearRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LinearRegression, self).__init__()
        self.linear= nn.Linear(n_input_features, 1)

    def forward(self, x):
        return self.linear(x)
model= nn.Linear(n_features, 1)
[w, b]= model.parameters()
print(f"w[0]= {w[0][0].item():.3f}, w[1]= {w[0][1].item():.3f}, w[2]= {w[0][2].item():.3f}, w[3]= {w[0][3].item():.3f}, b= {b.item():.3f}")
# print(w.shape)
learning_rate= 0.01
n_iters= 100
loss= nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(), lr= learning_rate)
J= []

def train():
    for epoch in range(n_iters):
        Y_pred= model(X_train)

        cost= loss(Y_pred, Y_train)
        cost.backward()
        J.append(cost.item())

        optimizer.step()
        optimizer.zero_grad()

        if (epoch+1) % 1 == 0:
            [w, b]= model.parameters()
            print(f"{epoch+1}, w[0]= {w[0][0].item():.3f}, w[1]= {w[0][1].item():.3f}, w[2]= {w[0][2].item():.3f}, w[3]= {w[0][3].item():.3f}, b= {b.item():.3f}, Cost= {cost.item():.3f}")

def plots():
    plt.plot(np.arange(n_iters), J)
    plt.show()

def metics():
    Y_pred= model.forward(X_test)
    r2= r2_score(Y_test, Y_pred.detach().numpy())
    print(f"R2= {r2}")

if __name__=="__main__":
    train()
    metics()
    plots()