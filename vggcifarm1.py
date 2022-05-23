import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time

import wandb
wandb.init(project="vggcifarm1")


device = torch.device("mps")
# device = torch.device("cpu")
print(device)
vgg = torchvision.models.vgg11(pretrained=True)
features = vgg.features
flat_features = nn.Flatten()
fc = nn.Linear(in_features=512, out_features=10)
model = nn.Sequential(features, flat_features, fc)

print(model)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epoch = 4
batch_size = 32
learning_rate = 0.001

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

n_total_steps = len(trainloader)
print("Start Training...")
start_time_train = time.time()
for epoch in range(num_epoch):
    start_time_epoch = time.time()
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for ii in range(labels.size(0)):
            label = labels[ii]
            pred = predicted[ii]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1
        
        # Log accuracy and training loss using wandb
        wandb.log({'Training Loss': loss.item(), 'Training Accuracy': n_correct/n_samples*100})
        
        # print training accuracy and loss per every 100 steps
        
        if (i+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epoch}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        
    print(f'Time taken for epoch {epoch+1} is {time.time() - start_time_epoch}')
    wandb.log({'Time taken for epoch': time.time() - start_time_epoch})
    

print(f'Total time taken for training is {time.time() - start_time_train}')