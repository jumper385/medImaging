#%%
import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import random

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

import torch 
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
#%%
def getList(dataPath):

    transformations = transforms.Compose([
        transforms.Resize(2000),
        transforms.CenterCrop(1800),
        transforms.ToTensor(),
    ])
    train_set = datasets.ImageFolder(f'{dataPath}/dataset', transform = transformations)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    
    return train_loader
trainloader = getList('./chest_xray')
x, y = next(iter(getList(f'./chest_xray')))
#%%
val = random.randint(0,31)
print(x[val][0].shape)
plt.imshow(x[val][1].data.reshape(100,100))
print(y[val].data)
#%%
class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 16, kernel_size= 2)
        # self.flatten1 = nn.Linear(16*24*24, 120)
        # self.fc1 = nn.Linear(120, 64)
        # self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16*24*24)
        # x = F.relu(self.flatten1(x))
        # x = F.relu(self.fc1(x))
        # x = F.softmax(self.fc2(x), dim=0)
        return x

net = Network()
net(x[0].reshape(1, 3 ,1800,1800))
#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

#%%
for epoch in range(2):
    running_loss = 0
    for i, data in tqdm(enumerate(trainloader, 0)):
        # print(len(data))
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i%10 == 9:
            print(running_loss/10)
            running_loss = 0.0

#%%


#%%
