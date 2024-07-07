import seaborn as sns
titanic_data = sns.load_dataset('titanic')
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# print(titanic_data.isna().sum())
# CHANGE "NA" VALUE IN AGE
titanic_data['age'].fillna(titanic_data['age'].mean(),inplace=True)
# print(titanic_data['age'].isna().sum())

# DROP ROWS WITH "NA" FROM COLUMNS
titanic_data.drop('deck',axis=1,inplace=True)
titanic_data = titanic_data.dropna(subset=['embark_town'])
titanic_data = titanic_data.dropna(subset=['embarked'])
# print(titanic_data.head())
# print(titanic_data.dtypes)

# MAP SEX 0:MALE, 1:FEMALE, ETC
# print(titanic_data['embark_town'].value_counts())
# print(titanic_data['embarked'].value_counts())
# print(titanic_data['sex'].value_counts())
titanic_data['sex_binary'] = titanic_data['sex'].map({'male': 0, 'female': 1})
titanic_data['embarked'] = titanic_data['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
titanic_data['class'] = titanic_data['class'].map({'First': 0, 'Second': 1, 'Third': 2})
titanic_data['embark_town'] = titanic_data['embark_town'].map({'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2})
titanic_data['adult_male'] = titanic_data['adult_male'].map({True: 0, False: 1})
# titanic_data['alone'] = titanic_data['alone'].map({True: 0, False: 1})
# for i in titanic_data:
#     print(i, titanic_data[i].isna().sum())
titanic_data.drop(['who','sex','alive'],axis=1,inplace=True)
# print(titanic_data['survived'].value_counts())
# print(titanic_data.head())
xy=titanic_data.to_numpy(dtype= np.float32)
X, Y= xy[:,1:], xy[:,0]
# print(X[0], Y[0])
n_samples, n_features= X.shape
sc= StandardScaler()
X= sc.fit_transform(X)
Y= Y.reshape(Y.shape[0], 1)
X= torch.from_numpy(X.astype(np.float32))
Y= torch.from_numpy(Y.astype(np.float32))
# print(X[:10], Y[:10])
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
n_iters= 800
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

        if epoch%50== 0:
            print(f"Epoch {epoch}: Loss= {l}")

def accuracy():
    Y_pred= model.forward(X_test)
    Y_pred_cls= Y_pred.round()
    acc= Y_pred_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    print(f"Accuracy= {acc}")

def plots():
    plt.plot(np.arange(n_iters), J, 'b')
    plt.show()

if __name__=="__main__":
    train()

    accuracy()

    plots()

    pass