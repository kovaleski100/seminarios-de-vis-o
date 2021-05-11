import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import ndimage


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



def applyTransformation(image, transf, val):
    if transf == 'rotate':
        image[0] = torch.tensor(ndimage.rotate(image[0], val, reshape=False)) # Red
        image[1] = torch.tensor(ndimage.rotate(image[1], val, reshape=False)) # Green
        image[2] = torch.tensor(ndimage.rotate(image[2], val, reshape=False)) # Blue
        # imshow(image)
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
    PATH = 'cifar_net.pth'
    

    testloader, classe, batch = load_dataset()
    applyTransf = True

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
            if applyTransf:
                for imgIdx in range(len(images)):
                    images[imgIdx] = applyTransformation(images[imgIdx+1], 'rotate', 45)
                    return
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