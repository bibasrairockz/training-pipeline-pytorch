import torch
import torch.nn as nn

X= torch.tensor([ [1], [2], [3], [4] ], dtype= torch.float32)
Y= torch.tensor([ [2], [4], [6], [8] ], dtype= torch.float32)
X_test= torch.tensor([5], dtype= torch.float32)
n_sample, n_features= X.shape
input_dim= output_dim= n_features
learning_rate= 0.01
n_iters= 100
model= nn.Linear(input_dim, output_dim)
loss= nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(), lr= learning_rate)

print(f"Before trainig: f(5)= {model(X_test)}")
print(f"Number of samples= {n_sample}\nNumber of features= {n_features}")

for epoch in range(n_iters):
    Y_pred= model(X)
    l= loss(Y, Y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch%10 == 0:
        [w, b]= model.parameters()
        print(f"Epoch {epoch}: w= {w[0][0].item():.3f}, loss= {l:.3f}")

print(f"After trainig: f(5)= {model(X_test)}")