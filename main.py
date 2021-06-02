import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

PATH = './cifar_net.pth'


# Creating CNN

class Net(nn.Module):

    def __init__(self, conv_layers, fc_layers):
        """
        init neural network with 2 conv and 2 fc.
        :param conv_layers: convolution layers of the second convolution
        :param fc_layers: count fully connected neuron.
        """
        self.fc_layers = fc_layers
        self.conv_layers = conv_layers
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, self.conv_layers, 5)
        self.fc1 = nn.Linear(self.conv_layers * 5 * 5, fc_layers)
        self.fc2 = nn.Linear(fc_layers, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        # do the LNN on data x with relu
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.conv_layers * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Creating linear CNN


class LinearNet(nn.Module):

    def __init__(self, conv_layers, fc_layers):
        # init neural network with 2 conv and 2 fc.
        self.fc_layers = fc_layers
        self.conv_layers = conv_layers
        super(LinearNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, self.conv_layers, 5)
        self.fc1 = nn.Linear(self.conv_layers * 5 * 5, fc_layers)
        self.fc2 = nn.Linear(fc_layers, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # do the LNN on data x

        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, self.conv_layers * 5 * 5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


colours = ['red', 'yellow', 'black', 'pink', 'purple']


def train_test_and_plot(net, name, permutation=None, kindOfNN="" ):
    """
    This function gets a nn and and trains and tests and plots the different value errors
    :param net: A neural Network
    :param name: A string
    :param permutation: boolean
    :param kindOfNN: A string
    :return:
    """
    # Creating NN
    # net = Net()
    train_loss = []
    test_loss = []
    # defining Optimizer and Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # Training the Data
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if permutation is not None:
                inputs = permutation(inputs)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        train_loss.append((running_loss / 48000))
        # test
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                if permutation is not None:
                    images = permutation(images)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_loss.append(1 - (correct / total))
    x = np.arange(1, 11)
    plt.plot(x, test_loss, linestyle='-', color=colours[int(name) - 1],
             label=kindOfNN+' test loss CV=' + str(net.conv_layers) + " FC=" + str(net.fc_layers))
    plt.plot(x, train_loss, linestyle='--', color=colours[int(name) - 1],
             label=kindOfNN+' train loss CV=' + str(net.conv_layers) + " FC=" + str(net.fc_layers))
    torch.save(net.state_dict(), PATH + name)


def train_test_and_plot_q4(net, name, permutation=None):
    """
    This function gets a nn and and trains and tests and plots the different value errors
    :param net: A neural Network
    :param name: A string
    :param permutation: boolean
    :return:
    """
    colours_q4 = ['red', 'yellow', 'black', 'pink', 'purple', 'brown', 'deepskyblue', 'maroon', 'violet', 'navy']

    test_loss = [[] for i in range(10)]
    # defining Optimizer and Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # Training the Data
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if permutation is not None:
                inputs = permutation(inputs)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        for i in range(10):
            test_loss[i].append(1 - class_correct[i] / class_total[i])

    x = np.arange(1, 11)
    for i in range(10):
        plt.plot(x, test_loss[i], linestyle='--', color=colours_q4[i], label='Test error rate of class: ' + classes[i])

    torch.save(net.state_dict(), PATH + name)


def question_1():
    """
    create and train five NN with difference in count of conv-layers and fully connectd layers.
    :return:
    """
    # First NN
    net1 = Net(16, 40)
    train_test_and_plot(net1, '1')
    # Second
    net2 = Net(200, 300)
    train_test_and_plot(net2,'2')
    # Third
    net3 = Net(10, 15)
    train_test_and_plot(net3,'3')
    # Four
    net4 = Net(100, 200)
    train_test_and_plot(net4,'4')
    # Five
    net5 = Net(20, 55)
    train_test_and_plot(net5,'5')
    plt.legend()
    plt.title("Over-fitting and under-fitting neural networks")
    plt.tight_layout()
    plt.savefig('question_1.png')
    plt.show()


def question_2():
    """
    Testing Non linear NN vs Linear NN
    :return: None
    """
    net1 = LinearNet(20, 50)
    train_test_and_plot(net1, '1', None, 'Linear Net with')
    net2 = LinearNet(20, 70)
    train_test_and_plot(net2, '2', None, 'Linear Net with')
    net3 = LinearNet(20, 100)
    train_test_and_plot(net3, '3', None, 'Linear Net with')
    net4 = Net(20, 50)
    train_test_and_plot(net4, '4', None, 'Non-Linear Net with')
    plt.legend()
    plt.title("Linear NN with different FC layers")
    plt.tight_layout()
    plt.savefig('question_2.png')
    plt.show()


def perm_function(data, perm):
    """
    A function that permutes a data by g given vector
    :param data: A batch of 4 pictures
    :param perm: A vector size of 1024
    :return: The permuted batch data
    """
    for k in range(4):
        for j in range(3):
            data_vec = data[k][j].reshape(1024)
            data_vec_perm = data_vec[perm]
            data[k][j] = data_vec_perm.reshape([32, 32])
    return data


def question_3():
    """
    Tests a nn with permuted data vs a nn with non permuted data
    :return:
    """
    perm = torch.randperm(1024)
    f_perm = lambda x: perm_function(x, perm)
    net = Net(16, 120)
    train_test_and_plot(net, '1', f_perm, 'NN with permuted data')
    net = Net(16,120)
    train_test_and_plot(net, '2', None, 'NN with normal data')
    plt.legend()
    plt.title("Impact of permuted data on training and testing neural network")
    plt.tight_layout()
    plt.savefig('question_3.png')
    plt.show()


def question_4():
    """
    A function that creates the question 4. permuting the data differently each batch
    :return: None
    """
    f_perm = lambda x: perm_function(x, torch.randperm(1024))
    net = Net(16, 120)
    train_test_and_plot_q4(net, 'q4', f_perm)
    plt.legend()
    plt.title("No Spatial Structure - question 4")
    plt.tight_layout()
    plt.savefig('question4.png')
    plt.show()


if __name__ == '__main__':
    # Loading dataset

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    question_1()
    question_2()
    question_3()
    question_4()

