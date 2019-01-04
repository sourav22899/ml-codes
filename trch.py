import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TwoLayerNet(nn.Module):
    def __init__(self,D_in,H,D_out):
        super(TwoLayerNet,self).__init__()
        self.fc1 = nn.Linear(D_in,H)
        self.fc2 = nn.Linear(H,D_out)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

N,D_in,H,D_out = 256,1024,200,10
X = torch.randn(N,D_in)
y = torch.randn(N,D_out)
model = TwoLayerNet(D_in,H,D_out)

criteria = nn.MSELoss()
opt = optim.Adam(model.parameters())

for epoch in range(500):
    y_pred = model(X)
    loss = criteria(y_pred,y)
    print(epoch,loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()