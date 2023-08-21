# -*- coding:utf-8 -*-
from torchvision.datasets.mnist import MNIST
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import custom_dataset

if __name__ == '__main__':

    training_data = np.load('data/kmnist-npz/kmnist-train-imgs.npz')['arr_0'] /255
    data_mean = training_data.mean()
    data_std = training_data.std()


    default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    #transformations 
    transform_train = transforms.Compose(
        [transforms.ToPILImage(), 
        transforms.Pad(4),
        transforms.RandomCrop(size = (28,28)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((data_mean,), (data_std,))
    ])

    transform_valid = transforms.Compose(
        [transforms.ToPILImage(), 
        transforms.ToTensor(),
        transforms.Normalize((data_mean,), (data_std,)),
        ])


    folder = './data/kmnist-npz/'

    train = custom_dataset(folder, train_or_test='train', transforms=transform_train)
    val = custom_dataset(folder, train_or_test='test', transforms=transform_valid)
    test = custom_dataset(folder, train_or_test='val', transforms=transform_valid)

    train_loader = DataLoader(train, batch_size=128, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, batch_size=128, shuffle=True,  pin_memory=True)
    test_loader = DataLoader(test, batch_size=128, shuffle=True, pin_memory=True)


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, 
                                stride=1, padding=2)
            self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1,
                                padding=1)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                                padding=1)
            self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                                padding=1)

            self.layer1 = nn.Sequential(self.conv1,nn.ReLU(), nn.MaxPool2d(kernel_size=2,stride= 2))
            self.layer2 = nn.Sequential(self.conv2,nn.ReLU(),  nn.MaxPool2d(kernel_size=2,stride = 2))
            self.layer3 = nn.Sequential(self.conv3,nn.ReLU(),  nn.MaxPool2d(kernel_size=2,stride = 2))
            self.layer4 = nn.Sequential(self.conv4,nn.ReLU(), nn.MaxPool2d(kernel_size=2,stride = 2))

            self.out = nn.Sequential(nn.Linear(2304, 1028),nn.ReLU(), nn.Linear(1028, 10)) # output 10 classes
        
        
        def forward(self, x):
            # x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            # x = self.layer4(x)

            x = x.view(x.size(0), -1) 
            x = self.out(x)
            m = nn.LogSoftmax(dim = 1)
            output = m(x)

            return output

    
    epochList = []
    LossFunctionList = []
    writer = SummaryWriter()
    
    def training_loop(epochs, model, loss_fn, optimizer, trainloader, val_loader):
        model.train()
        total_step = len(train_loader)
        for epoch in range(epochs): #loop over the dataset multiple times
            running_loss = 0.0
            correct, total = 0,0
            for i, data in enumerate(trainloader, 0):
                #get the inputs; data is a list of [inputs,labels]
                imgs, labels = data
                imgs,labels = imgs.to(default_device),labels.to(default_device)
                
                #zero the parameter gradients
                optimizer.zero_grad()

                #fowards + backward + optimize
                output = model(imgs)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()

                writer.add_scalar("Loss/train",loss,epoch)

                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                print ('{} Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(datetime.datetime.now(), epoch +1, epochs, 
                            i + 1, total_step, loss.item()))

            accuracy = 100* correct// total 
            writer.add_scalar("Accuracy/Train",accuracy,epoch)                   
            
            #Save model at the end of each epoch
            PATH = './checkpoints.cifar_net_{:02d}.pth'.format(epoch)
            torch.save(model.state_dict(),PATH)

            #Validation 
            validation_loss = 0.0
            running_loss = 0.0
            print('Starting validation for epoch {}'.format(epoch+1))
            model.eval()
            with torch.no_grad():
                correct, total = 0, 0
                for data in val_loader:
                    images, labels = data
                    images, labels = images.to(default_device),labels.to(default_device)

                    output = model(images)
                    loss = loss_fn(output, labels)
                    validation_loss += loss.item()

                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accuracy = 100* correct// total
                print(f'Accuracy of the network on the testing images: {100* correct// total} %')

                validation_loss /=len(val_loader)
                writer.add_scalar("Loss/Val",validation_loss,epoch)
                writer.add_scalar("Accuracy/Val",accuracy,epoch)
                epochList.append(epoch+1)
                LossFunctionList.append(validation_loss)
                print('Validation loss for epoch {:2d}: {:5f}'.format(epoch+1,validation_loss))
        print('Finished Training')

        


    model = Net()
    model.to(default_device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3,betas = (0.9,0.999))
    loss_fn = nn.CrossEntropyLoss()

    training_loop(
        epochs = 10,
        model = model,
        loss_fn = loss_fn,
        optimizer = optimiser,
        trainloader = train_loader,
        val_loader = val_loader
    )


    #Gets the best training file for using on the validation set
    print (LossFunctionList)
    print(epochList)
    index_min = np.argmin(LossFunctionList)
    print(epochList[index_min])
    if (index_min< 10):
        base_filename = 'checkpoints.cifar_net_0{}.pth'
    else:
        base_filename = 'checkpoints.cifar_net_{}.pth'
    base_filename = base_filename.format(index_min)
    print(base_filename)

    #Testing
    net = Net()
    net.to(default_device)
    net.load_state_dict(torch.load(base_filename))
    correct = 0
    total = 0
    print("starting validation")
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(default_device),labels.to(default_device)
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the validation images: {100* correct// total} %')


