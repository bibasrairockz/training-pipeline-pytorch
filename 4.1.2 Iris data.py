from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import datasets
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


iris= datasets.load_iris()
X_numpy, Y_numpy= iris.data, iris.target
# print(iris.feature_names)
# print(X_numpy.shape, type(Y_numpy[0]))
n_samples, n_features= X_numpy.shape
sc= StandardScaler()
# X_numpy= sc.fit_transform(X_numpy)
Y_numpy= Y_numpy.reshape(Y_numpy.shape[0], 1)
X, Y= torch.from_numpy(X_numpy.astype(np.float32)), torch.from_numpy(Y_numpy.astype(np.float32))
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.15, random_state= 1)
# print(X_train[0], Y_train.shape)
class LinearRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LinearRegression, self).__init__()
        self.linear= nn.Linear(n_input_features, 1)

    def forward(self, x):
        return self.linear(x)
model= nn.Linear(n_features, 1)
w, b= model.parameters()
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

        if (epoch+1) % 2 == 0:
            [w, b]= model.parameters()
            print(f"{epoch+1}, w[0]= {w[0][0].item():.3f}, w[1]= {w[0][1].item():.3f}, w[2]= {w[0][2].item():.3f}, w[3]= {w[0][3].item():.3f}, b= {b.item():.3f}, Cost= {cost.item():.5f}")

def plots():
    plt.plot(np.arange(n_iters), J)
    plt.xlabel("iters")
    plt.ylabel("J")
    plt.show()

def accuracy():
    Y_pred= model.forward(X_test)
    # print(Y_pred.shape, Y_pred[:20])
    Y_pred_cls= Y_pred.round()
    # print(Y_pred_cls.shape, Y_pred_cls[:20])
    acc= Y_pred_cls.eq(Y_test).sum()/float(Y_test.shape[0])
    print(f"Accuracy= {acc}") 

def plotdata():
    _, ax = plt.subplots()
    scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
    ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
    _ = ax.legend(
        scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
    )
    plt.show()

if __name__== "__main__":
    # plotdata()
    train()
    accuracy()
    # plots()
    pass