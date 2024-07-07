import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

bm= pd.read_csv('./data/bank marketing/bank-full.csv', delimiter=';')
bm['job']= bm['job'].map({'management': 0, 'technician': 1, 'entrepreneur': 2, 'blue-collar': 3, 'unknown': 4, \
                         'retired': 5, 'admin.': 6, 'services': 7, 'self-employed':8, 'unemployed': 9, \
                        'housemaid': 10, 'student': 11})
bm['marital']= bm['marital'].map({'married': 0, 'single': 1, 'divorced': 2})
bm['education']= bm['education'].map({'tertiary': 0, 'secondary': 1, 'unknown': 2, 'primary': 3})
bm['default']= bm['default'].map({'no': 0, 'yes': 1})
bm['housing']= bm['housing'].map({'no': 0, 'yes': 1})
bm['loan']= bm['loan'].map({'no': 0, 'yes': 1})
bm['contact']= bm['contact'].map({'unknown': 0, 'cellular': 1, 'telephone': 2})
bm['month']= bm['month'].map({'may': 0, 'jun': 1, 'jul': 2, 'aug': 3, 'oct': 4, 'nov': 5, 'dec': 6,\
                                   'jan': 7, 'feb': 8, 'mar': 9, 'apr': 10, 'sep': 11})
bm['poutcome']= bm['poutcome'].map({'unknown': 0, 'failure': 1,  'other': 2, 'success': 3})
bm['y']= bm['y'].map({'no': 0, 'yes': 1})
xy=bm.to_numpy(dtype= np.float32)
X, Y= xy[:,0:16], xy[:,16]
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
n_iters= 500
loss= nn.BCELoss()
optimizer= torch.optim.SGD(model.parameters(), lr= learning_rate)
J= []

def train():
    for epoch in range(n_iters):
        Y_pred= model.forward(X_train)

        l= loss(Y_pred, Y_train)
        l.backward()
        J.append(l.item())

        optimizer.step()
        optimizer.zero_grad()

        if epoch%10 == 0:
            print(f"Epoch {epoch}: Cost= {l}")

def accuracy():
    Y_pred= model.forward(X_test)
    Y_pred_cls= Y_pred.round()
    acc= Y_pred_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    print(f"Accuracy= {acc}")

def plots():
    plt.plot(np.arange(n_iters), J, 'b')
    plt.xlabel("iters")
    plt.ylabel("J")
    plt.show()

if __name__=="__main__":
    train()

    accuracy()

    plots()

    pass