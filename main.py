from __future__ import print_function
import argparse
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from utils.config_utils import read_args, load_config, Dict2Object


# initial function
class Net(nn.Module):
    def __init__(self):
        # self = this(in java)
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 把1维转换成32
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        # 你需要用到的东西的初始化定义？self是你这个定义的对象，。后是方法？
        # find in conv2D torch, calculating function

    def forward(self, x):
        # pass information from input layer to output layer
        # relu: y = x (x != 0)
        # x = images
        # 输入层
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # 输出的激活函数
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """
    train the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    # epoch: 训练多少次
    # input x; f(x), f(f(x))
    model.train()
    loss_total = 0
    batch_correct = 0  # 每batch正确的个数
    # 总数
    correct = 0
    loss = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # batch_loss_total = F.nll_loss(output, target).item()
        # loss += batch_loss_total * data.size(0)
        # predict = output.argmax(dim=1, keepdim=True)
        predict = (torch.max(output.data, 1))[1]
        correct += (predict == target).sum().item()  # The prediction is correct if it is the same as the target
        # correct += batch_correct
        # batch_size = data.size(0)
        # total += batch_size
        total += target.size(0)

        #
        # 每10次打印一下，保存损失
        #
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item(), 100. * correct / total))
            loss_total += loss.item()

    training_acc, training_loss = 100. * correct / total, loss_total / len(train_loader)

    return training_acc, training_loss


def test(model, device, test_loader):
    """
    test the model and return the testing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    # get epochs
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 不会对结果自动求导，节省内存
        for data, target in test_loader:
            #
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            predict = output.argmax(1, keepdim=True)
            correct += predict.eq(target.view_as(predict)).sum().item()

            # print the test results
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_loss /= len(test_loader.dataset)
    testing_acc, testing_loss = 100. * correct / len(test_loader.dataset), test_loss
    return testing_acc, testing_loss

# plot the charts of the train and test function
def plot(epoches, performance, yLabel, seed):
    """
    plot the model performance
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    """Fill your code"""
    plt.plot(epoches, performance, color='blue')
    plt.xlabel('epoch')
    plt.ylabel(yLabel)
    plt.title(f'seed = {seed}')
    plt.show()

# calling the functions above
def run(config, seed):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True, }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader, pls modify this function"""
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    """record the performance"""
    epoches = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch)
        """record training info, Fill your code"""
        training_accuracies.append(train_acc)
        training_loss.append(train_loss)

        test_acc, test_loss = test(model, device, test_loader)
        """record testing info, Fill your code"""
        testing_accuracies.append(test_acc)
        testing_loss.append(test_loss)

        print('epoch is', epoch, 'train accuracy is ', train_acc, ' the train loss is ', train_loss)
        epoches.append(epoch)
        scheduler.step()
        """update the records, Fill your code"""
        torch.save(model.state_dict(), './model.pth')
        torch.save(optimizer.state_dict(), './optimizer.pth')

    """plotting training performance with the records"""
    plot(epoches, training_accuracies, "train-Accuracy", seed)
    plot(epoches, training_loss, "train-Loss", seed)

    """plotting testing performance with the records"""
    plot(epoches, testing_accuracies, "test-Accuracy", seed)
    plot(epoches, testing_loss, "test-Loss", seed)

# save in txt files
    with open(f'txt/training-acc {seed}.txt', 'a') as f:
        f.writelines(str(training_accuracies))
    with open(f'txt/training-loss {seed}.txt', 'a') as f:
        f.writelines(str(training_loss))
    with open(f'txt/testing-acc {seed}.txt', 'a') as f:
        f.writelines(str(testing_accuracies))
    with open(f'txt/testing-loss {seed}.txt', 'a') as f:
        f.writelines(str(testing_loss))

    # df = pd.DataFrame(training_loss, columns=["epoch"])
    # df.to_csv(f'training-loss {seed}.csv', index=False)
    # df = pd.DataFrame(training_accuracies, columns=["epoch"])
    # df.to_csv(f'training-acc {seed}.csv', index=False)
    #
    # df = pd.DataFrame(testing_loss, columns=["epoch"])
    # df.to_csv(f'testing-loss {seed}.csv', index=False)
    # df = pd.DataFrame(testing_accuracies, columns=["epoch"])
    # df.to_csv(f'testing-acc {seed}.csv', index=False)

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


# function for drawing graphs of three seeds
# four graph for training-loss training-acc testing-loss and testing-acc
# calculate the average each of training-loss training-acc testing-loss and testing-acc
def plot_mean():
    """
    Read the recorded results.
    Plot the mean results after three runs.
    :return:
    """
    trl = np.array([])
    trl1 = np.array([])
    trl2 = np.array([])
    trl3 = np.array([])
    f = open('txt/training-loss 123.txt', 'r')
    line = f.readline()
    while line:
        data = eval(line)
        trl1 = np.append(trl1, data)
        line = f.readline()
    f2 = open('txt/training-loss 321.txt', 'r')
    line2 = f2.readline()
    while line2:
        data2 = eval(line2)
        trl2 = np.append(trl2, data2)
        line2 = f2.readline()
    f3 = open('txt/training-loss 666.txt', 'r')
    line3 = f3.readline()
    while line3:
        data3 = eval(line3)
        trl3 = np.append(trl3, data3)
        line3 = f3.readline()
    for j in range(0, 15):
        trl = np.append(trl, ((trl1[j] + trl2[j] + trl3[j])/3))
    x_value = list(range(15))
    plt.plot(x_value, trl, color='blue')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("training loss")
    plt.show()

    tra = np.array([])
    tra1 = np.array([])
    tra2 = np.array([])
    tra3 = np.array([])
    f = open('txt/training-acc 123.txt', 'r')
    line = f.readline()
    while line:
        data = eval(line)
        tra1 = np.append(tra1, data)
        line = f.readline()
    f2 = open('txt/training-acc 321.txt', 'r')
    line2 = f2.readline()
    while line2:
        data2 = eval(line2)
        tra2 = np.append(tra2, data2)
        line2 = f2.readline()
    f3 = open('txt/training-acc 666.txt', 'r')
    line3 = f3.readline()
    while line3:
        data3 = eval(line3)
        tra3 = np.append(tra3, data3)
        line3 = f3.readline()
    for j in range(0, 15):
        tra = np.append(tra, ((tra1[j] + tra2[j] + tra3[j]) / 3))
    x_value = list(range(15))
    plt.plot(x_value, tra, color='blue')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title("training accuracy")
    plt.show()

    tel = np.array([])
    tel1 = np.array([])
    tel2 = np.array([])
    tel3 = np.array([])
    f = open('txt/testing-loss 123.txt', 'r')
    line = f.readline()
    while line:
        data = eval(line)
        tel1 = np.append(tel1, data)
        line = f.readline()
    f2 = open('txt/testing-loss 321.txt', 'r')
    line2 = f2.readline()
    while line2:
        data2 = eval(line2)
        tel2 = np.append(tel2, data2)
        line2 = f2.readline()
    f3 = open('txt/testing-loss 666.txt', 'r')
    line3 = f3.readline()
    while line3:
        data3 = eval(line3)
        tel3 = np.append(tel3, data3)
        line3 = f3.readline()
    for j in range(0, 15):
        tel = np.append(tel, ((tel1[j] + tel2[j] + tel3[j]) / 3))
    x_value = list(range(15))
    plt.plot(x_value, tel, color='blue')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("testing loss")
    plt.show()

    tea = np.array([])
    tea1 = np.array([])
    tea2 = np.array([])
    tea3 = np.array([])
    f = open('txt/testing-acc 123.txt', 'r')
    line = f.readline()
    while line:
        data = eval(line)
        tea1 = np.append(tea1, data)
        line = f.readline()
    f2 = open('txt/testing-acc 321.txt', 'r')
    line2 = f2.readline()
    while line2:
        data2 = eval(line2)
        tea2 = np.append(tea2, data2)
        line2 = f2.readline()
    f3 = open('txt/testing-acc 666.txt', 'r')
    line3 = f3.readline()
    while line3:
        data3 = eval(line3)
        tea3 = np.append(tea3, data3)
        line3 = f3.readline()
    for j in range(0, 15):
        tea = np.append(tea, ((tea1[j] + tea2[j] + tea3[j]) / 3))
    x_value = list(range(15))
    plt.plot(x_value, tea, color='blue')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title("testing accuracy")
    plt.show()


# main in java
if __name__ == '__main__':
    arg = read_args()

    """toad training settings"""
    config = load_config(arg)

    """train model and record results"""

    p1 = multiprocessing.Process(target=run, args=(config, 123))
    p2 = multiprocessing.Process(target=run, args=(config, 321))
    p3 = multiprocessing.Process(target=run, args=(config, 666))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()

    """plot the mean results"""
    plot_mean()
