from natthaphon import Model
from models import ResNet
from torch.optim import SGD, lr_scheduler
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
import os
import math
import json
from torch import nn
import torchvision.transforms as transforms
import numpy as np
import torch


class Loader(DataLoader):
    def __len__(self):
        return int(round(len(self.dataset) / self.batch_size))


def lrlambda(t, m=6, T=781*300):
    alpha = (0.05/2)*(math.cos(math.pi*((t-1) % (T/m))/(T/m))+1)
    return alpha if alpha > 0 else 0.2


def lrstep(epoch):
    if epoch < 150:
        a = 0.05
    elif 150 < epoch < 225:
        a = 0.005
    else:
        a = 0.0005
    print(f'Epoch: {epoch+1} - returning learning rate {a}')
    return a


def decaylr(epoch):
    lr = 0.045 * 0.94 ** (2*int(epoch/2))
    return lr


class LambdaLR:
    def __init__(self, optim, lambda_fn):
        self.optimizer = optim
        self.lambda_fn = lambda_fn
        self.epoch = 0
        self.last_epoch = -1

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group in self.optimizer.param_groups:
            lr = self.lambda_fn(self.last_epoch)
            param_group['lr'] = lr


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


class SparceBCELoss(nn.Module):
    def forward(self, y_pred, y_true):
        y_true = to_categorical(y_true.long().cpu().detach().numpy(), 10)
        y_true = torch.from_numpy(y_true).cuda().float()
        y_pred = nn.Sigmoid()(y_pred.float())
        return nn.BCELoss()(y_pred, y_true)


if __name__ == '__main__':
    try:
        os.listdir('/root')
        rootpath = '/root/palm/DATA/'
    except PermissionError:
        rootpath = '/home/palm/PycharmProjects/DATA/'
    name = 'cifar10'
    root = os.path.join(rootpath, name)
    model = Model(ResNet())
    sgd = SGD(model.model.parameters(), 0.01, 0.9)
    model.compile(optimizer=sgd,
                  loss=SparceBCELoss(),
                  metric=model.categorical_accuracy(),
                  device='cuda'
                  )
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10(root=root, train=True, download=True, transform=transform_train)
    trainloader = Loader(trainset, batch_size=64, shuffle=True, num_workers=0)

    testset = CIFAR10(root=root, train=False, download=True, transform=transform_test)
    testloader = Loader(testset, batch_size=100, shuffle=False, num_workers=0)

    schedule = LambdaLR(sgd, lrstep)

    history = model.fit_generator(trainloader, 300, validation_data=testloader, schedule=schedule)
    with open('logs/misc-01.json', 'w') as wr:
        json.dump(history, wr)
