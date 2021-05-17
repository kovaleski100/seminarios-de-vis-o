import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import ndimage
from random import randint
import cv2
from numba import jit, cuda
import warnings

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes, batch_size


def saltPepper(self, image, ratio):
    height = len(image[0])
    width = len(image[0][0])
    numPixels = int(ratio*width*height)
    for i in range(numPixels):
        color = randint(0,1) * 2 - 1
        x = randint(0,width-1)
        y = randint(0,height-1)
        image[0][y][x] = color
        image[1][y][x] = color
        image[2][y][x] = color
    return image

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        print(len(x[0]))
        x = self.pool(F.relu(self.conv1(x)))
        print(len(x[0]))
        x = self.pool(F.relu(self.conv2(x)))
        print(len(x[0]))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        print(len(x[0]))
        x = F.relu(self.fc1(x))
        print(len(x[0]))
        x = F.relu(self.fc2(x))
        print(len(x[0]))
        x = self.fc3(x)
        return x

    


    def applyTransformation(self, image, transf, val=None):
        if transf == 'rotate':
            image[0] = torch.tensor(ndimage.rotate(image[0], val, reshape=False)) # Red
            image[1] = torch.tensor(ndimage.rotate(image[1], val, reshape=False)) # Green
            image[2] = torch.tensor(ndimage.rotate(image[2], val, reshape=False)) # Blue
        elif transf == 'shift':
            image[0] = torch.tensor(ndimage.shift(image[0], val, mode='constant'))
            image[1] = torch.tensor(ndimage.shift(image[1], val, mode='constant'))
            image[2] = torch.tensor(ndimage.shift(image[2], val, mode='constant'))
        elif transf == 'scale':
            # Buggy
            height = len(image[0])
            width = len(image[0][0])
            image0 = ndimage.zoom(image[0], val, mode='constant')
            image0 = cv2.resize(image0, (width, height))
            image[0] = torch.tensor(image0)
            image1 = ndimage.zoom(image[1], val, mode='constant')
            image1 = cv2.resize(image1, (width, height))
            image[1] = torch.tensor(image1)
            image2 = ndimage.zoom(image[2], val, mode='constant')
            image2 = cv2.resize(image2, (width, height))
            image[2] = torch.tensor(image2)
        elif transf == 'saltpepper':
            image = self.saltPepper(image, val)
        elif transf == 'negative':
            image = image / 2 + 0.5 # Unnormalize
            image = 1.0 - image # Apply inverse
            image = 2 * image - 1 # Normalize
        return image


def main():
    print("Hello World!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    train, test, classes, batch = load_dataset()

    net = Net()

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    X = list()
    for epoch in range(20):  # loop over the dataset multiple times
        print(epoch)
        running_loss = 0.0
        for i, data in enumerate(train, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            #print("none")
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #for imgIdx in range(len(inputs)):
            #    inputs[imgIdx] = net.applyTransformation(inputs[imgIdx], 'negative')

 #           outputs = net(inputs)
  #          loss = criterion(outputs, labels)
   #         loss.backward()
    #        optimizer.step()

            #for imgIdx in range(len(inputs)):
            #    inputs[imgIdx] = net.applyTransformation(inputs[imgIdx], 'saltpepper', 0.1) # Between 0 and 1

            #outputs = net(inputs)
            #loss = criterion(outputs, labels)
            #loss.backward()
            #optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net20_sp.pth'
    torch.save(net.state_dict(), PATH)




if __name__ == "__main__":
    main()
