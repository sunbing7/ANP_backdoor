import os
import time
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchsummary import summary
import torch.nn.functional as F
import models
import data.poison_cifar as poison

from data.data_loader import get_custom_cifar_loader, get_data_class_loader
from models.selector import *
import matplotlib.pyplot as plt
import copy

parser = argparse.ArgumentParser(description='Semantic backdoor mitigation.')

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
parser.add_argument('--poison-type', type=str, default='badnets', choices=['badnets', 'blend', 'clean-label', 'benign', 'semantic'],
                    help='type of backdoor attacks used during training')
parser.add_argument('--poison-rate', type=float, default=0.05,
                    help='proportion of poison examples in the training set')
parser.add_argument('--poison_target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')

parser.add_argument('--in_model', type=str, required=True, help='input model')
parser.add_argument('--t_attack', type=str, default='green', help='attacked type')
parser.add_argument('--data_name', type=str, default='CIFAR10', help='name of dataset')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--resume', type=int, default=1, help='resume from args.checkpoint')
parser.add_argument('--option', type=str, default='detect', choices=['detect', 'remove'], help='run option')
parser.add_argument('--lr', type=float, default=0.1, help='lr')
parser.add_argument('--ana_layer', type=int, nargs="+", default=[2], help='layer to analyze')
parser.add_argument('--num_sample', type=int, default=192, help='number of samples')
parser.add_argument('--plot', type=int, default=0, help='plot hidden neuron causal attribution')

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

    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.
    train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader = \
        get_custom_cifar_loader(args.data_dir, args.batch_size, args.poison_target, args.t_attack, 2500)

    # Step 1: create poisoned / clean dataset
    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader

    # Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(models, args.arch)(num_classes=10).to(device)

    state_dict = torch.load(args.in_model, map_location=device)
    load_state_dict(net, orig_state_dict=state_dict)

    summary(net, (3, 32, 32))
    #print(net)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)
    '''
    # Step 3: train backdoored models
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_init.th'))
    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    print('0 \t None     \t None     \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))
    '''
    # analyze hidden neurons
    for each_class in range (0, args.num_class):
        print('Analyzing class:{}.'.format(each_class))
        analyze_eachclass(net, args.arch, each_class, args.num_class, args.num_sample, args.ana_layer, plot=args.plot)

    return
    '''
    for epoch in range(1, args.epoch):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train(model=net, criterion=criterion, optimizer=optimizer,
                                      data_loader=train_mix_loader)

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
    '''


def analyze_eachclass(model, model_name, cur_class, num_class, num_sample, ana_layer, plot=False):
    '''
    use samples from base class, find important neurons
    '''
    clean_class_loader = get_data_class_loader(args.data_dir, args.batch_size, cur_class, args.t_attack)
    hidden_test = analyze_hidden(model, model_name, clean_class_loader, cur_class, num_sample, ana_layer)

    if plot:
        hidden_test_all = []
        hidden_test_name = []
        for this_class in range(0, num_class):
            hidden_test_all_ = []
            for i in range(0, len(ana_layer)):
                temp = hidden_test[i][:, [0, (this_class + 1)]]
                hidden_test_all_.append(temp)

            hidden_test_all.append(hidden_test_all_)

            hidden_test_name.append('class' + str(this_class))

        plot_multiple(hidden_test_all, hidden_test_name, cur_class, ana_layer, save_n="test")


def analyze_hidden(model, model_name, class_loader, cur_class, num_sample, ana_layer):
    out = []
    for cur_layer in ana_layer:
        print('current layer: {}'.format(cur_layer))
        model1, model2 = split_model(model, model_name, num_sample, split_layer=cur_layer)
        model1.eval()
        model2.eval()
        #summary(model1, (3, 32, 32))
        #summary(model2, (128, 16, 16))
        out = []
        do_predict_avg = []
        total_num_samples = 0
        for image, gt in class_loader:
            if total_num_samples >= num_sample:
                break

            image, gt = image.to(device), gt.to(device)

            # compute output
            with torch.no_grad():
                dense_output = model1(image)
                #dense_output = dense_output.permute(0, 2, 3, 1)
                ori_output = model2(dense_output)

                dense_hidden_ = torch.clone(torch.reshape(dense_output, (dense_output.shape[0], -1)))
                #ori_output = filter_model(image)
                do_predict_neu = []
                do_predict = []
                #do convention for each neuron
                for i in range(0, len(dense_hidden_[0])):
                    hidden_do = np.zeros(shape=dense_hidden_[:, i].shape)
                    dense_output_ = torch.clone(dense_hidden_)
                    dense_output_[:, i] = torch.from_numpy(hidden_do)
                    dense_output_ = torch.reshape(dense_output_, dense_output.shape)
                    dense_output_ = dense_output_.to(device)
                    output_do = model2(dense_output_).cpu().detach().numpy()
                    do_predict_neu.append(output_do) # 4096x32x10
                do_predict_neu = np.array(do_predict_neu)
                do_predict_neu = np.abs(ori_output.cpu().detach().numpy() - do_predict_neu)
                do_predict = np.mean(np.array(do_predict_neu), axis=1)  #4096x10

            do_predict_avg.append(do_predict) #batchx4096x11
            total_num_samples += len(gt)
        # average of all baches
        do_predict_avg = np.mean(np.array(do_predict_avg), axis=0) #4096x10
        # insert neuron index
        idx = np.arange(0, len(do_predict_avg), 1, dtype=int)
        do_predict_avg = np.c_[idx, do_predict_avg]
        #out = do_predict_avg[:, [0, (target_class + 1)]]
        out.append(do_predict_avg)
        np.savetxt(args.output-dir + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt",
                   do_predict_avg, fmt="%s")

    return np.array(out)


def plot_multiple(_rank, name, cur_class, ana_layer, normalise=False, save_n=""):
    # plot the permutation of cmv img and test imgs
    plt_row = len(_rank)

    rank = []
    for _rank_i in _rank:
        rank.append(copy.deepcopy(_rank_i))

    plt_col = len(ana_layer)
    fig, ax = plt.subplots(plt_row, plt_col, figsize=(7 * plt_col, 5 * plt_row), sharex=False, sharey=True)

    col = 0
    for do_layer in ana_layer:
        for row in range(0, plt_row):
            # plot ACE
            if row == 0:
                ax[row, col].set_title('Layer_' + str(do_layer))
                # ax[row, col].set_xlabel('neuron index')
                # ax[row, col].set_ylabel('delta y')

            if row == (plt_row - 1):
                # ax[row, col].set_title('Layer_' + str(do_layer))
                ax[row, col].set_xlabel('neuron index')

            ax[row, col].set_ylabel(name[row])

            # Baseline is np.mean(expectation_do_x)
            if normalise:
                rank[row][col][:, 1] = rank[row][col][:, 1] / np.max(rank[row][col][:, 1])

            ax[row, col].scatter(rank[row][col][:, 0].astype(int), rank[row][col][:, 1], label=str(do_layer) + '_cmv',
                                 color='b')
            ax[row, col].legend()

        col = col + 1
    if normalise:
        plt.savefig(args.output-dir + "plt_n_c" + str(cur_class) + save_n + ".png")
    else:
        plt.savefig(args.output-dir + "plt_c" + str(cur_class) + save_n + ".png")
    # plt.show()


def split_model(ori_model, model_name, split_layer=4):
    '''
    split given model from the dense layer before logits
    Args:
        ori_model:
        model_name: model name
    Returns:
        splitted models: 2-5
    '''
    if model_name == 'resnet18':
        modules = list(ori_model.children())
        module1 = modules[:2]
        module2 = modules[2:split_layer]
        module3 = modules[split_layer:6]
        module4 = [modules[6]]

        model_1st = nn.Sequential(*[*module1, Relu(), *module2])
        model_2nd = nn.Sequential(*[*module3, Avgpool2d(), Flatten(), *module4])

        '''
        layers = [modules[0]] + [modules[1]] + list(modules[2]) + list(modules[3]) + list(modules[4]) + list(modules[5]) + [modules[6]]
        module1 = layers[:split_layer]
        module2 = layers[split_layer:(len(layers) - 1)]
        module3 = [layers[-1]]
        model_1st = nn.Sequential(*module1)
        model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])
        '''
    else:
        return None, None

    return model_1st, model_2nd


class Relu(nn.Module):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, x):
        x = F.relu(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Avgpool2d(nn.Module):
    def __init__(self):
        super(Avgpool2d, self).__init__()

    def forward(self, x):
        x = F.avg_pool2d(x, 4)
        return x

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


def _adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 21:
        lr = lr
    elif epoch < 100:
        lr = 0.1 * lr
    else:
        lr = 0.0009
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
    if args.option == 'detect':
        main()

