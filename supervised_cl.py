import argparse
from collections import defaultdict
import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


import torchvision

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

CLASSES_PER_TASK = 10

import torch
from PIL import Image
from torchvision import transforms


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # readout head
        self.readout = nn.Sequential(nn.Linear(2048, 512), nn.BatchNorm1d(512),
                               nn.ReLU(), nn.Linear(512, 100, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        readout = self.readout(feature)
        return F.normalize(feature, dim=-1), readout


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    criterion = nn.CrossEntropyLoss()
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for data_tuple in train_bar:
        data, target = data_tuple
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        feature, readout = net(data)
        loss = criterion(readout, target)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size
        train_bar.set_description(
            'Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, args.epochs, total_loss / total_num))
    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, test_data_loader, name):
    net.eval()
    total_top1, total_top5, total_num = 0.0, 0.0, 0.0
    with torch.no_grad():
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data_tuple in test_bar:
            data, target = data_tuple
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, readout = net(data)
            pred_labels = readout.argsort(dim=-1, descending=True)
            total_num += data.size(0)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('{} Test Epoch: re[{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(name, epoch, args.epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))
    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
    # args parse
    args = parser.parse_args()

    cifar_transform = transforms.Compose([
                                        transforms.RandomResizedCrop(32),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    model = Model(args.feature_dim).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    cifar100_train = torchvision.datasets.CIFAR100(root='data', train=True, transform=cifar_transform, download=True)
    cifar100_test = torchvision.datasets.CIFAR100(root='data', train=False, transform=cifar_transform, download=True)
    n_tasks = int(len(cifar100_train.classes) / CLASSES_PER_TASK)
    past_test_loaders = [None] * n_tasks

    for task in range(n_tasks):
        target_class = set(range(task * CLASSES_PER_TASK, (task + 1) * CLASSES_PER_TASK))
        train_data = torch.utils.data.Subset(cifar100_train, [i for i, t in enumerate(cifar100_train.targets) if t in target_class])
        test_data = torch.utils.data.Subset(cifar100_test, [i for i, t in enumerate(cifar100_test.targets) if t in target_class])

        train_data.classes = list(target_class)
        test_data.classes = list(target_class)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                drop_last=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
        past_test_loaders[task] = test_loader

        # training loop
        results = defaultdict(list)

        if not os.path.exists('results'):
            os.mkdir('results')
        best_acc = 0.0
        test_period = 5
        for epoch in range(1, args.epochs + 1):
            train_loss = train(model, train_loader, optimizer)
            if epoch % test_period == 0:
                results['train_loss'].append(train_loss)
                for t, test_loader in enumerate(past_test_loaders):
                    if t > task:
                        break
                    test_acc_1, test_acc_5 = test(model, test_loader, "task_%i" % t)
                    results['task_%i_test_acc@1' % t].append(test_acc_1)
                    results['task_%i_test_acc@5' % t].append(test_acc_5)
                    if t == task and test_acc_1 > best_acc:
                        best_acc = test_acc_1
                        torch.save(model.state_dict(), 'results/model.pth')
                # save statistics
                data_frame = pd.DataFrame(data=results,
                                          index=((np.arange(len(results['train_loss'])) + 1) * test_period).tolist())
                data_frame.to_csv('results/statistics.csv', index_label='epoch')
            if epoch % 50 == 0:
                torch.save(model.state_dict(), 'results/model_{}.pth'.format(epoch))
