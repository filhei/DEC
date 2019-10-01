import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.optim as optim
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import sys

class DEC():
    def __init__(self, num_points=320, n_dim=4, range=1, mini_batch_size=16):
        self.num_points = num_points
        self.n_dim = n_dim
        self.range = range
        self.mini_batch_size = mini_batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = "cpu"


    def generate_points(self):
        points = np.random.uniform(-self.range, self.range,
                                   (self.num_points, self.n_dim))
        classes = np.zeros((self.num_points, 1))
        for i in range(self.num_points):
            for j in range(self.n_dim):
                classes[i] += np.power(2,j) * (points[i,j] > 0)

        return points, classes


    def load_mnist(self):
        transform = transforms.ToTensor()
        trainset = datasets.MNIST('MNIST_data/train', download=False, train=True, transform=transform)
        valset = datasets.MNIST('MNIST_data/val', download=False, train=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.mini_batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=self.mini_batch_size, shuffle=True)

        return trainloader, valloader


    def load_cifar(self):
        transform = transforms.ToTensor()
        trainset = datasets.CIFAR10(root='./CIFAR_data', train=True,
                                            download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.mini_batch_size,
                                                  shuffle=True, num_workers=2)
        testset = datasets.CIFAR10(root='./CIFAR_data', train=False,
                                               download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.mini_batch_size,
                                                shuffle=False, num_workers=2)
        classes = ('plane', 'car', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return trainloader, testloader, classes


    def compute_loss(self, encodings, classes, beta=0.5):
        loss = 0
        #distance_to_origin_loss = 0
        range_loss = 0

        num_points = encodings.size()[0]
        for i in range(num_points):
            range_loss += (torch.norm(encodings[i]) - 1.).pow(2)
            #distance_to_origin_loss += torch.norm(encodings[i])
            for j in range(num_points):
                if j != i:
                    sign = 1. if classes[i] == classes[j] else -1.
                    sign = Variable(torch.tensor(sign), requires_grad=True).to(self.device)
                    loss += sign * torch.dist(encodings[i], encodings[j], 2)

        loss /= (num_points * (num_points - 1))  #, requires_grad=True)
        #distance_to_origin_loss /=  num_points
        range_loss /=  num_points
        total_loss = loss + beta*range_loss
        return total_loss


    def train(self, epochs = 10, learn_rate=1e-3, mini_batch_size=16):
        self.model = CNN(output_dim=2).to(self.device)
        trainloader, valloader, _ = self.load_cifar()
        dataiter = iter(trainloader)
        sample_images, sample_labels = dataiter.next()
        sample_images = sample_images.to(self.device)
        sample_labels = sample_labels.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learn_rate)
        num_mini_batches = 50000 // mini_batch_size
        print("=========== START TRAINING ===========")
        for epoch in range(epochs):
            epoch_loss = 0
            count = 0

            for images, labels in trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.compute_loss(outputs, labels)
                epoch_loss += loss.detach()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                count += 1
                self.drawProgressBar(epoch, count/num_mini_batches, (loss/count).item())

            print("")
            loss /= count
            #print("Epoch: " + str(epoch) + ",   loss: " + str(loss.item()))
            self.save_model()
            #self.scatter_plot(sample_images, sample_labels)

        self.save_model()


    def test_model(self, path="saved_models/cifar_temp_model", ):
        self.model = CNN(output_dim=2).to(self.device)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        trainloader, valloader, _ = self.load_cifar()
        dataiter = iter(valloader)
        images, labels = dataiter.next()
        #sort after class
        labels, indices = labels.sort()
        images = images[indices]

        images = images.to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            outputs = self.model(images)

        self.evalutation_plot(images.cpu(), outputs.cpu(), labels.cpu())


    def scatter_plot(self, images, classes):
        with torch.no_grad():
            encodings = self.model(images)
        plt.clf()
        plt.scatter(encodings[:,0].cpu(), encodings[:,1].cpu(), c=classes.squeeze().cpu())
        plt.pause(0.01)


    def evalutation_plot(self, images, encodings, labels):
        figure1 = plt.figure()
        num_of_images = self.mini_batch_size
        for index in range(1, self.mini_batch_size+1):
            plt.subplot(8, self.mini_batch_size // 8, index)
            plt.axis('off')
            plt.imshow(images[index-1].numpy().transpose((1,2,0))) #, cmap='gray_r')

        figure2 = plt.figure()
        plt.scatter(encodings[:,0], encodings[:,1], c=labels.squeeze())
        for index in range(0, self.mini_batch_size):
            x = encodings[index, 0]
            y = encodings[index, 1]
            plt.text(x * (1 + 0.1), y * (1 + 0.1), str(index+1), fontsize=10)
        plt.show()

    def drawProgressBar(self, epoch, percent, avg_loss, barLen = 20):
        sys.stdout.write("\r")
        progress = ""
        for i in range(barLen):
            if i < int(barLen * percent):
                progress += "="
            else:
                progress += " "
        sys.stdout.write("Epoch %d: |%s| %.2f%% - loss: %.5f" % (epoch, progress, percent * 100, avg_loss))
        sys.stdout.flush()


    def save_model(self, path="saved_models/cifar_temp_model"):
        torch.save(self.model.state_dict(), path)


class CNN(nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 7, 1)
        self.conv2 = nn.Conv2d(16, 32, 5, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(7*7*64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# =============================== Main ======================================= #
if __name__ == "__main__":
    train = False
    if train == True:
        mini_batch_size = 32
        model = DEC(mini_batch_size=mini_batch_size)
        model.train(mini_batch_size=mini_batch_size, epochs=20)
    else:
        mini_batch_size = 32
        model = DEC(mini_batch_size=mini_batch_size)
        model.test_model()
