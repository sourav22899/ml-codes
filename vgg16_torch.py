import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_set = datasets.CIFAR10(root = './cifardata',train = True,download = True,
                transform = transform)
test_set = datasets.CIFAR10(root = './cifardata',train = False,download = True,
                transform = transform)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# classes = ('plane,...')
train_loader = torch.utils.data.DataLoader(dataset = train_set,batch_size = 32,
                shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_set,batch_size = 32,
                shuffle = True)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size = 3,padding = 1,stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size = 2,stride = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size = 3,padding = 1,stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size = 2,stride = 2)
        )
        self.fc = nn.Linear(8*8*64,10)
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape(0),-1)
        x = self.fc(x)
        return (x)

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

########### HyperParameters ###########
num_epochs = 100
total_step = len(train_loader)


for epoch in range(num_epochs):
    for i,(image,label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        output = model(image)
        loss = criterion(output,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))