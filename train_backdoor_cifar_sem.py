import os
import time
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import models
import data.poison_cifar as poison

from data.data_loader import get_custom_cifar_loader
from models.selector import *

parser = argparse.ArgumentParser(description='Train poisoned networks')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn'])
parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--epoch', type=int, default=200, help='the numbe of epoch for training')
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--save-every', type=int, default=20, help='save checkpoints every few epochs')
parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--output-dir', type=str, default='logs/models/')
# backdoor parameters
parser.add_argument('--clb-dir', type=str, default='', help='dir to training data under clean label attack')
parser.add_argument('--poison-type', type=str, default='badnets', choices=['badnets', 'blend', 'clean-label', 'benign'],
                    help='type of backdoor attacks used during training')
parser.add_argument('--poison-rate', type=float, default=0.05,
                    help='proportion of poison examples in the training set')
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.
    train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader = \
        get_custom_cifar_loader(args.data_dir, args.batch_size, args.poison_target, args.t_attack, 500)

    # Step 1: create poisoned / clean dataset
    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader

    # Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(models, args.arch)(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    # Step 3: train backdoored models
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_init.th'))

    for epoch in range(1, args.epoch):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train(model=net, criterion=criterion, optimizer=optimizer,
                                      data_loader=train_mix_loader)
        train_loss2, train_sr = train(model=net, criterion=criterion, optimizer=optimizer,
                                      data_loader=train_adv_loader)
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        scheduler.step()
        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc)

        if (epoch + 1) % args.save_every == 0:
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_{}.th'.format(epoch)))

    # save the last checkpoint
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_last.th'))


def train(model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


if __name__ == '__main__':
    main()
