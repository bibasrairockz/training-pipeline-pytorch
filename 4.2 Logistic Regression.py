import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

bc= datasets.load_breast_cancer()
X, Y= bc.data, bc.target
n_samples, n_features= X.shape
sc= StandardScaler()
X= sc.fit_transform(X)
Y= Y.reshape(Y.shape[0], 1)
X= torch.from_numpy(X.astype(np.float32))
Y= torch.from_numpy(Y.astype(np.float32))
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.2, random_state=1234)
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear= nn.Linear(n_input_features, 1)

    def forward(self, x):
        Y_pred= torch.sigmoid(self.linear(x))
        return Y_pred
model= LogisticRegression(n_features)
learning_rate= 0.01
n_iters= 100
loss= nn.BCELoss()
optimizer= torch.optim.SGD(model.parameters(), lr= learning_rate)

def train():
    for epoch in range(n_iters):
        Y_pred= model.forward(X_train)

        l= loss(Y_pred, Y_train)
        l.backward()

        optimizer.step()
        optimizer.zero_grad()

        if epoch%10== 0:
            print(f"Epoch {epoch}: Loss= {l}")

def accuracy():
    Y_pred= model.forward(X_test)
    Y_pred_cls= Y_pred.round()
    acc= Y_pred_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    print(f"Accuracy= {acc}")

if __name__=="__main__":
    train()

    accuracy()

