import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

BATCH_SIZE = 32

# Define the Network Structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)  # Flatten the input image (28x28) and then Fully Connected Layer of 10 neurons
        self.fc2 = nn.Linear(10, 10)     # Second layer with 10 neurons
        self.fc3 = nn.Linear(10, 10)     # Third Layer with 10 output neurons

    def forward(self, x):
        x = x.view(-1, 28*28)    # Flatten the input
        x = torch.sigmoid(self.fc1(x))  # Apply activation function
        x = torch.sigmoid(self.fc2(x))  # Apply activation function
        x = self.fc3(x)  # No activation function in the last layer, softmax will be applied in the loss function
        return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(),  # Convert to Tensor
                                transforms.Normalize((0.1307,), (0.3081,))])  # Normalize input

train_data = MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
test_data = MNIST(root='./mnist_data/', train=False, transform=transform)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=1000, shuffle=False)

# print train_data
for i in range(3):
    print(train_data[i][1])

    for j in range(28):
        for k in range(28):
            if train_data[i][0][0][j][k] > 0.5:
                print(1, end='')
            else:
                print(0, end='')
        print()



net = Net()  
criterion = nn.CrossEntropyLoss()  

lr = 0.01  # learning rate

# Training the Model
for epoch in range(10):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        net.zero_grad()  # zeros the gradient buffers of all parameters
        outputs = net(inputs) 
        loss = criterion(outputs, labels)
        loss.backward()  # backpropagation
        
        # manually update parameters
        with torch.no_grad():
            for param in net.parameters():
                param -= lr * param.grad

        print(f'Epoch: {epoch+1}, Batch: {i+1}, Loss: {loss.item()}, Processed Images: {(i+1)*BATCH_SIZE} / {len(train_data)}')


# Define network, loss function, and optimizer
# net = Net()
# criterion = nn.CrossEntropyLoss()  # This applies LogSoftmax internally
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# # Training the Model
# for epoch in range(10):  # can change the number of epochs
#     for i, data in enumerate(train_loader):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

# Testing the Model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')