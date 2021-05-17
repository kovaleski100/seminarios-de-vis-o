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


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    batch_size = 32

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
    return testloader, classes, batch_size


def saltPepper(image, ratio):
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


def applyTransformation(image, transf, val=None):
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
        image = saltPepper(image, val)
    elif transf == 'negative':
        image = image / 2 + 0.5 # Unnormalize
        image = 1.0 - image # Apply inverse
        image = 2 * image - 1 # Normalize
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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    PATH = 'cifar_net20_neg+sp.pth'
    

    testloader, classe, batch = load_dataset()

    dataiter = iter(testloader)
    # images, labels = dataiter.next()
    # print('GroundTruth: ', ' '.join('%5s' % classe[labels[j]] for j in range(4)))
    net = Net()

    # net.load_state_dict(torch.load(PATH))
    net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

    # outputs = net(images)
    # _, predicted = torch.max(outputs, 1)
    # print('Predicted: ', ' '.join('%5s' % classe[predicted[j]]
    #                           for j in range(4)))

    correct_pred = {classname: 0 for classname in classe}
    total_pred = {classname: 0 for classname in classe}
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            for imgIdx in range(len(images)):
                images[imgIdx] = applyTransformation(images[imgIdx], '')
                # images[imgIdx] = applyTransformation(images[imgIdx], 'rotate', 30)
                # images[imgIdx] = applyTransformation(images[imgIdx], 'negative')
                # images[imgIdx] = applyTransformation(images[imgIdx], 'shift', [0, 10]) # Horizontal shift
                # images[imgIdx] = applyTransformation(images[imgIdx], 'scale', 2)
                # images[imgIdx] = applyTransformation(images[imgIdx], 'saltpepper', 0.1) # Between 0 and 1
                # imshow(images[imgIdx])
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classe[label]] += 1
                total_pred[classe[label]] += 1
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,accuracy))


if __name__ == '__main__':
    main()
