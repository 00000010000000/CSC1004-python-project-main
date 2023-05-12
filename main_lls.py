from __future__ import print_function
import argparse
import multiprocessing
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

        batch_loss_total = F.nll_loss(output, target, reduction='sum').item()
        loss += batch_loss_total
        predict = output.argmax(dim=1, keepdim=True)
        batch_correct += (predict == target).sum().item()  # 预测和target一样则为正确
        # correct += batch_correct
        # batch_size = data.size(0)
        # total += batch_size
        total = len(train_loader.dataset)
    '''Fill your code'''
    #
    #  每10次打印一下，保存损失
    #     if batch_idx % args.log_interval == 0:
    #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             epoch, batch_idx * len(data),len(train_loader.dataset),
    #             100. * batch_idx / len(train_loader), loss.item()))
    #
    training_acc, training_loss = 100. * correct / total, loss / total
    # with open("training.txt","a") as f:
    #     f.write(f"train: epoch {epoch} | loss: {training_loss:.4f} | accuracy: {training_acc:.2f}\n")
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
            '''Fill your code'''
            #
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            # test_loss /= len(test_loader.dataset)
            # test_losses.append(test_loss)
            # print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
            #
        # with open("testing.txt", 'a') as f:
        #     f.write(f"Testing accuracy: {(100. * correct / len(test_loader.dataset)):.2f}%, "
        #             f"Testing loss: {testing_loss:.4f}\n")
    testing_acc, testing_loss = 100. * correct / len(test_loader.dataset), test_loss / len(
        test_loader.dataset)
    return testing_acc, testing_loss


def plot(epoches, performance, yLabel, seed):
    """
    plot the model performance
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    """Fill your code"""
    plt.plot(epoches, performance, color='blue')
    plt.xlabel(epoches)
    plt.ylabel(yLabel)
    plt.title(f'seed = {seed}')
    plt.show()


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

    """plotting training performance with the records"""

    plot(epoches, training_accuracies, "train-Accuracy", seed)
    plot(epoches, training_loss, "train-Loss", seed)

    df = pd.DataFrame(training_loss, columns=["epoch"])
    df.to_csv(f'training-loss {seed}.csv', index=False)
    df = pd.DataFrame(training_accuracies, columns=["epoch"])
    df.to_csv(f'training-acc {seed}.csv', index=False)

    """plotting testing performance with the records"""
    plot(epoches, testing_accuracies, "test-Accuracy", seed)
    plot(epoches, testing_loss, "test-Loss", seed)

    df = pd.DataFrame(testing_loss, columns=["epoch"])
    df.to_csv(f'testing-loss {seed}.csv', index=False)
    df = pd.DataFrame(testing_accuracies, columns=["epoch"])
    df.to_csv(f'testing-acc {seed}.csv', index=False)

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
    """fill your code"""
    # sl = [123, 321, 666]
    trl = []
    # for seed in sl:
    #     data1 = pd.read_csv(f'training-loss {seed}.csv')
    #     for i in range(len(data1)):
    #         trl[i] = data1
    #         l[seed] += trl[i]
    data11 = pd.read_csv(f'csv/training-loss 123.csv')
    data12 = pd.read_csv(f'csv/training-loss 321.csv')
    data13 = pd.read_csv(f'csv/training-loss 666.csv')
    for j in range(0, 15):
        trl[j] = (data11[j] + data12[j] + data13[j])/3
    # for seed in sl:
    #     data1 = pd.read_csv(f'training-loss {seed}.csv').sum()
    #     trl.append(data1)
    # data1 = pd.read_csv(f'training-loss 123.csv').sum()
    # data2 = pd.read_csv(f'training-loss 321.csv').sum()
    # data3 = pd.read_csv(f'training-loss 666.csv').sum()
    # trl.append(data1)
    # trl.append(data2)
    # trl.append(data3)
    plt.plot((0, 15), trl, color='blue')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("training loss")
    plt.show()

    tra = []
    data21 = pd.read_csv(f'csv/training-acc 123.csv')
    data22 = pd.read_csv(f'csv/training-acc 321.csv')
    data23 = pd.read_csv(f'csv/training-acc 666.csv')
    for j in range(0, 15):
        tra[j] = (data21[j] + data22[j] + data23[j]) / 3
    # for seed in sl:
    #     data2 = pd.read_csv(f'training-acc {seed}.csv').sum()
    #     tra.append(data2)

    # data5 = pd.read_csv(f'training-acc 123.csv').sum()
    # data6 = pd.read_csv(f'training-acc 321.csv').sum()
    # data7 = pd.read_csv(f'training-acc 666.csv').sum()
    # tra.append(data5)
    # tra.append(data6)
    # tra.append(data7)
    plt.plot((0, 15), tra, color='blue')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title("training accuracy")
    plt.show()

    tel = []
    data31 = pd.read_csv(f'csv/testing-loss 123.csv')
    data32 = pd.read_csv(f'csv/testing-loss 321.csv')
    data33 = pd.read_csv(f'csv/testing-loss 666.csv')
    for j in range(0, 15):
        tel[j] = (data31[j] + data32[j] + data33[j]) / 3
    # for seed in sl:
    #     data3 = pd.read_csv(f'testing-loss {seed}.csv').sum()
    #     tel.append(data3)

    # data1 = pd.read_csv(f'testing-loss 123.csv').sum()
    # data2 = pd.read_csv(f'testing-loss 321.csv').sum()
    # data3 = pd.read_csv(f'testing-loss 666.csv').sum()
    # tel.append(data1)
    # tel.append(data2)
    # tel.append(data3)
    plt.plot((0, 15), tel, color='blue')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("testing loss")
    plt.show()

    tea = []
    data41 = pd.read_csv(f'csv/testing-acc 123.csv')
    data42 = pd.read_csv(f'csv/testing-acc 321.csv')
    data43 = pd.read_csv(f'csv/testing-acc 666.csv')
    for j in range(0, 15):
        tea[j] = (data41[j] + data42[j] + data43[j]) / 3
    # for seed in sl:
    #     data4 = pd.read_csv(f'testing-acc {seed}.csv').sum()
    #     tea.append(data4)

    # data8 = pd.read_csv(f'testing-acc 123.csv').sum()
    # data9 = pd.read_csv(f'testing-acc 321.csv').sum()
    # data10 = pd.read_csv(f'testing-acc 666.csv').sum()
    # tra.append(data8)
    # tra.append(data9)
    # tra.append(data10)
    plt.plot((0, 15), tea, color='blue')
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
    #p2 = multiprocessing.Process(target=run, args=(config, 321))
    #p3 = multiprocessing.Process(target=run, args=(config, 666))

    p1.start()
    #p2.start()
    #p3.start()

    p1.join()
    #p2.join()
    #p3.join()

    """plot the mean results"""
    plot_mean()
