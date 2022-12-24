import os
import time
import argparse
import logging
import numpy as np
import torch
from torchsummary import summary
import torch.nn.functional as F
import models

from data.data_loader import get_custom_loader, get_custom_class_loader, get_data_adv_loader, get_dataset_info
from models.selector import *
import matplotlib.pyplot as plt
import copy
from collections import Counter
from models.split_model import split_model, reconstruct_model, recover_model
from models.gen_trigger import UAP

parser = argparse.ArgumentParser(description='Semantic backdoor mitigation.')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn', 'vgg11_bn'])
parser.add_argument('--widen_factor', type=int, default=1, help='widen_factor for WideResNet')
parser.add_argument('--batch_size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--epoch', type=int, default=200, help='the numbe of epoch for training')
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--save_every', type=int, default=20, help='save checkpoints every few epochs')
parser.add_argument('--data_dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--output_dir', type=str, default='logs/models/')
# backdoor parameters
parser.add_argument('--clb_dir', type=str, default='', help='dir to training data under clean label attack')
parser.add_argument('--poison_type', type=str, default='badnets', choices=['badnets', 'blend', 'clean-label', 'benign', 'semantic'],
                    help='type of backdoor attacks used during training')
parser.add_argument('--poison-rate', type=float, default=0.05,
                    help='proportion of poison examples in the training set')
parser.add_argument('--poison_target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger_alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')

parser.add_argument('--in_model', type=str, required=True, help='input model')
parser.add_argument('--t_attack', type=str, default='green', help='attacked type')
parser.add_argument('--data_name', type=str, default='CIFAR10', help='name of dataset')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--resume', type=int, default=1, help='resume from args.checkpoint')
parser.add_argument('--option', type=str, default='detect', choices=['detect', 'remove', 'plot', 'causality_analysis',
                                                                     'remove2', 'gen_trigger'], help='run option')
parser.add_argument('--lr', type=float, default=0.1, help='lr')
parser.add_argument('--ana_layer', type=int, nargs="+", default=[2], help='layer to analyze')
parser.add_argument('--num_sample', type=int, default=192, help='number of samples')
parser.add_argument('--plot', type=int, default=0, help='plot hidden neuron causal attribution')
parser.add_argument('--reanalyze', type=int, default=0, help='redo analyzing')
parser.add_argument('--confidence', type=int, default=2, help='detection confidence')
parser.add_argument('--potential_source', type=int, default=0, help='potential source class of backdoor attack')
parser.add_argument('--reg', type=float, default=0.9, help='trigger generation reg factor')

args = parser.parse_args()
args_dict = vars(args)
#print(args_dict)
state = {k: v for k, v in args._get_kwargs()}
for key, value in state.items():
    print("{} : {}".format(key, value))
os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def causality_analysis():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ])
    #logger.info(args)

    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.
    train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader = \
        get_custom_loader(args.data_dir, args.batch_size, args.poison_target, args.data_name, args.t_attack, 2500)

    # Step 1: create poisoned / clean dataset
    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader

    # Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(models, args.arch)(num_classes=args.num_class).to(device)

    state_dict = torch.load(args.in_model, map_location=device)
    load_state_dict(net, orig_state_dict=state_dict)

    #summary(net, (3, 32, 32))
    #print(net)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)
    #'''
    logger.info('Epoch \t lr \t Time \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_init.th'))
    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    logger.info('0 \t None \t None \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))
    #'''
    # analyze hidden neurons
    if args.reanalyze:
        for each_class in range (0, args.num_class):
            print('Analyzing class:{}'.format(each_class))
            analyze_eachclass(net, args.arch, each_class, args.num_class, args.num_sample, args.ana_layer, plot=args.plot)

    return


def detect():
    # Step 1 find target class
    if args.reanalyze:
        analyze_pcc(args.num_class, args.ana_layer)
    flag_list = detect_pcc(args.num_class)
    #print('pcc flag list: {}'.format(flag_list))
    potential_target = flag_list[-1][0]

    #'''
    #'''
    # Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(models, args.arch)(num_classes=args.num_class).to(device)

    state_dict = torch.load(args.in_model, map_location=device)
    load_state_dict(net, orig_state_dict=state_dict)

    #summary(net, (3, 32, 32))
    #print(net)

    #criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    flag_list = analyze_source_class2(net, args.arch, args.poison_target, potential_target, args.num_class, args.ana_layer, args.num_sample)

    print('[Detection] potiental source class: {}, target class: {}'.format(int(flag_list), int(potential_target)))

    #'''
    return


def remove_exp():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ])
    #logger.info(args)

    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.
    train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader = \
        get_custom_loader(args.data_dir, args.batch_size, args.poison_target, args.data_name, args.t_attack, 2500)

    adv_class_loader = get_data_adv_loader(args.data_dir, args.batch_size, args.poison_target, args.data_name,
                                                args.t_attack)

    # Step 1: create poisoned / clean dataset
    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader

    # Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(models, args.arch)(num_classes=args.num_class).to(device)

    state_dict = torch.load(args.in_model, map_location=device)
    load_state_dict(net, orig_state_dict=state_dict)

    #summary(net, (3, 32, 32))
    #print(net)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)
    #'''
    # Step 3: train backdoored models
    logger.info('Epoch \t lr \t Time \t CleanLoss \t CleanACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_init.th'))
    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    logger.info('0 \t None \t None \t None \t None \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))
    #'''
    for epoch in range(1, args.epoch):
        start = time.time()
        _adjust_learning_rate(optimizer, epoch, args.lr)
        lr = optimizer.param_groups[0]['lr']
        #train_loss, train_acc = train_tune(model=net, criterion=criterion, optimizer=optimizer,
        #                              data_loader=train_clean_loader, adv_loader=adv_class_loader)

        #train_loss, train_acc = train(model=net, criterion=criterion, optimizer=optimizer,
        #                              data_loader=adv_class_loader)

        train_loss, train_acc = train(model=net, criterion=criterion, optimizer=optimizer,
                                      data_loader=train_clean_loader)

        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        scheduler.step()
        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc)

        if (epoch + 1) % args.save_every == 0:
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_finetune_{}_{}.th'.format(args.t_attack, epoch)))

    # save the last checkpoint
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_finetune_' + str(args.t_attack) + '_last.th'))
    #'''

    return


def remove_exp2():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ])
    #logger.info(args)

    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.
    train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader = \
        get_custom_loader(args.data_dir, args.batch_size, args.poison_target, args.data_name, args.t_attack, 2500)

    #adv_class_loader = get_data_adv_loader(args.data_dir, args.batch_size, args.poison_target, args.data_name,
    #                                            args.t_attack)

    # Step 1: create poisoned / clean dataset
    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader

    # Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(models, args.arch)(num_classes=args.num_class).to(device)

    state_dict = torch.load(args.in_model, map_location=device)
    load_state_dict(net, orig_state_dict=state_dict)
    mask = np.zeros(512)
    neu_idx = np.loadtxt(args.output_dir + "/outstanding_" + "c" + str(1) + "_target_" + str(args.poison_target) + ".txt")
    mask[neu_idx.astype(int)] = 1
    mask = torch.from_numpy(mask).to(device)
    net = reconstruct_model(net, args.arch, mask, split_layer=args.ana_layer[0])

    #summary(net, (3, 32, 32))
    #print(net)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)
    #'''
    # Step 3: train backdoored models
    logger.info('Epoch \t lr \t Time \t CleanLoss \t CleanACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_init.th'))
    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    logger.info('0 \t None \t None \t None \t         None \t         {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))
    #'''
    for epoch in range(1, args.epoch):
        start = time.time()
        _adjust_learning_rate(optimizer, epoch, args.lr)
        lr = optimizer.param_groups[0]['lr']
        #train_loss, train_acc = train_tune(model=net, criterion=criterion, optimizer=optimizer,
        #                              data_loader=train_clean_loader, adv_loader=adv_class_loader)

        #train_loss, train_acc = train(model=net, criterion=criterion, optimizer=optimizer,
        #                              data_loader=adv_class_loader)

        train_loss, train_acc = train(model=net, criterion=criterion, optimizer=optimizer,
                                      data_loader=train_clean_loader)

        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        scheduler.step()
        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc)

        if (epoch + 1) % args.save_every == 0:
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_finetune_{}_{}.th'.format(args.t_attack, epoch)))

    net = recover_model(net, args.arch, split_layer=args.ana_layer[0])

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)
    #'''
    # Step 3: train backdoored models
    #logger.info('Epoch \t lr \t Time \t CleanLoss \t CleanACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    #torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_init.th'))
    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    logger.info('0 \t None \t None \t None \t         None \t         {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))
    #'''

    # save the last checkpoint
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_finetune_' + str(args.t_attack) + '_last.th'))
    #'''

    return


def gen_trigger_test():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ])
    #logger.info(args)

    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.
    train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader = \
        get_custom_loader(args.data_dir, args.batch_size, args.poison_target, args.data_name, args.t_attack, 2500)

    clean_class_loader = get_custom_class_loader(args.data_dir, args.batch_size, args.potential_source, args.data_name,
                                                 args.t_attack)

    # Step 1: create poisoned / clean dataset
    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader

    # Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(models, args.arch)(num_classes=args.num_class).to(device)

    state_dict = torch.load(args.in_model, map_location=device)
    load_state_dict(net, orig_state_dict=state_dict)

    #summary(net, (3, 32, 32))
    #print(net)
    dshape, dchn, dmean, dstd = get_dataset_info(args.data_name)

    generator = UAP(shape=dshape,
                num_channels=dchn,
                mean=dmean,
                std=dstd,
                device=device)

    net = nn.Sequential(OrderedDict([('generator', generator), ('target_model', net)]))
    net = torch.nn.DataParallel(net)

    net.module.target_model.eval()
    net.module.generator.train()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)
    '''
    logger.info('Epoch \t lr \t Time \t CleanLoss \t CleanACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    logger.info('0 \t None \t None \t None \t None \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))
    '''
    count = 0
    for i, (images, _) in enumerate(clean_class_loader):
        for image in images:
            if count >= args.num_sample:
                break
            for epoch in range(1, int(args.epoch / args.batch_size)):
                start = time.time()
                _adjust_learning_rate(optimizer, epoch, args.lr)
                lr = optimizer.param_groups[0]['lr']

                train_loss, train_acc = train_trigger(model=net, criterion=criterion, optimizer=optimizer,
                                              target_class=args.poison_target, image=image, batch_size=args.batch_size, reg=args.reg)

                #cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
                #po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
                scheduler.step()
                end = time.time()
                logger.info(
                    '%d \t %.3f \t %.1f \t %.4f \t %.4f',
                    epoch, lr, end - start, train_loss / epoch, train_acc / (epoch * args.batch_size))

                if (epoch + 1) % args.save_every == 0:
                    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_trigger_{}_{}.th'.format(args.t_attack, epoch)))

            # save the last checkpoint
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_trigger_' + str(args.t_attack) + '_last.th'))

            # export mask
            mask = torch.unsqueeze(generator.uap, dim=0)
            mask = mask[0].cpu().detach().numpy()

            image = image.cpu().detach().numpy()
            mask = np.transpose(mask, (1, 2, 0))
            image = np.transpose(image, (1, 2, 0))
            # Put image into original form
            orig_img = image * dstd + dmean

            # Add uap to input
            adv_orig_img = orig_img + mask
            # Put image into normalized form
            adv_x = (adv_orig_img - dmean) / dstd

            plot_mask = adv_x
            plot_tuap_normal = plot_mask + 0.5
            plot_tuap_amp = plot_mask / 2 + 0.5
            tuap_range = np.max(plot_tuap_amp) - np.min(plot_tuap_amp)
            plot_tuap_amp = plot_tuap_amp / tuap_range + 0.5
            plot_tuap_amp -= np.min(plot_tuap_amp)
            imgplot = plt.imshow(plot_tuap_amp)
            plt.savefig(os.path.join(args.output_dir, 'model_trigger_mask_' + str(args.t_attack) + '_' + str(count) + '.png'))

            np.save(os.path.join(args.output_dir, 'model_trigger_mask_' + str(args.t_attack) + '_' + str(count) + '.npy'), mask)
            count = count + 1
    return


def gen_trigger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ])
    #logger.info(args)

    if args.poison_type != 'semantic':
        print('Invalid poison type!')
        return

    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.
    train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader = \
        get_custom_loader(args.data_dir, args.batch_size, args.poison_target, args.data_name, args.t_attack, 2500)

    clean_class_loader = get_custom_class_loader(args.data_dir, args.batch_size, args.potential_source, args.data_name,
                                                 args.t_attack)

    # Step 1: create poisoned / clean dataset
    poison_test_loader = test_adv_loader
    clean_test_loader = test_clean_loader


    # Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    ori_net = getattr(models, args.arch)(num_classes=args.num_class).to(device)

    state_dict = torch.load(args.in_model, map_location=device)
    load_state_dict(ori_net, orig_state_dict=state_dict)

    #summary(net, (3, 32, 32))
    #print(net)
    dshape, dchn, dmean, dstd = get_dataset_info(args.data_name)


    #for all samples
    count = 0
    for i, (images, _) in enumerate(clean_class_loader):
        for image in images:
            if count >= args.num_sample:
                break

            generator = UAP(shape=dshape,
                        num_channels=dchn,
                        mean=dmean,
                        std=dstd,
                        device=device)

            net = nn.Sequential(OrderedDict([('generator', generator), ('target_model', ori_net)]))
            net = torch.nn.DataParallel(net)

            net.module.target_model.eval()
            net.module.generator.train()

            criterion = torch.nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

            for epoch in range(0, int(args.epoch / args.batch_size)):
                start = time.time()
                _adjust_learning_rate(optimizer, epoch, args.lr)
                lr = optimizer.param_groups[0]['lr']

                train_loss, train_acc = train_trigger(model=net, criterion=criterion, optimizer=optimizer,
                                              target_class=args.poison_target, image=image, batch_size=args.batch_size, reg=args.reg)

                #cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
                #po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
                scheduler.step()
                end = time.time()
                logger.info(
                    '%d \t %.3f \t %.1f \t %.4f \t %.4f',
                    epoch, lr, end - start, train_loss / (epoch + 1), train_acc / ((epoch + 1) * args.batch_size))

                #if (epoch + 1) % args.save_every == 0:
                #    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_trigger_{}_{}.th'.format(args.t_attack, epoch)))

            # save the last checkpoint
            #torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_trigger_' + str(args.t_attack) + '_last.th'))

            # export mask
            mask = torch.unsqueeze(generator.uap, dim=0)
            mask = mask[0].cpu().detach().numpy()

            image = image.cpu().detach().numpy()
            mask = np.transpose(mask, (1, 2, 0))
            image = np.transpose(image, (1, 2, 0))
            # Put image into original form
            orig_img = image * dstd + dmean

            # Add uap to input
            adv_orig_img = orig_img + mask
            # Put image into normalized form
            adv_x = (adv_orig_img - dmean) / dstd

            plot_mask = adv_x
            plot_tuap_normal = plot_mask + 0.5
            plot_tuap_amp = plot_mask / 2 + 0.5
            tuap_range = np.max(plot_tuap_amp) - np.min(plot_tuap_amp)
            plot_tuap_amp = plot_tuap_amp / tuap_range + 0.5
            plot_tuap_amp -= np.min(plot_tuap_amp)
            imgplot = plt.imshow(plot_tuap_amp)
            plt.savefig(os.path.join(args.output_dir, 'model_trigger_mask_' + str(args.t_attack) + '_' + str(count) + '.png'))

            np.save(os.path.join(args.output_dir, 'model_trigger_mask_' + str(args.t_attack) + '_' + str(count) + '.npy'), mask)
            count = count + 1

            del generator, net, mask, criterion, optimizer, scheduler
    return


def hidden_plot():
    for each_class in range (0, args.num_class):
        print('Plotting class:{}'.format(each_class))
        hidden_test_all = []
        hidden_test_name = []
        for this_class in range(0, args.num_class):
            hidden_test_all_ = []
            for i in range(0, len(args.ana_layer)):
                hidden_test = np.loadtxt(
                    args.output_dir + "/test_pre0_" + "c" + str(this_class) + "_layer_" + str(args.ana_layer[i]) + ".txt")
                temp = hidden_test[:, [0, (this_class + 1)]]
                hidden_test_all_.append(temp)

            hidden_test_all.append(hidden_test_all_)

            hidden_test_name.append('class' + str(this_class))

        plot_multiple(hidden_test_all, hidden_test_name, each_class, args.ana_layer, save_n="test")


def analyze_eachclass(model, model_name, cur_class, num_class, num_sample, ana_layer, plot=False):
    '''
    use samples from base class, find important neurons
    '''
    clean_class_loader = get_custom_class_loader(args.data_dir, args.batch_size, cur_class, args.data_name, args.t_attack)
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


def analyze_advclass(model, model_name, cur_class, num_class, num_sample, ana_layer, plot=False):
    '''
    use samples from base class, find important neurons
    '''
    adv_class_loader = get_data_adv_loader(args.data_dir, args.batch_size, args.poison_target, args.data_name, args.t_attack)
    hidden_test = analyze_hidden(model, model_name, adv_class_loader, cur_class, num_sample, ana_layer)

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
        model1, model2 = split_model(model, model_name, split_layer=cur_layer)
        model1.eval()
        model2.eval()
        #summary(model1, (3, 32, 32))
        #summary(model2, (128, 16, 16))
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
                old_output = model(image)
                dense_hidden_ = torch.clone(torch.reshape(dense_output, (dense_output.shape[0], -1)))
                #ori_output = filter_model(image)
                do_predict_neu = []
                do_predict = []
                #do convention for each neuron
                for i in range(0, len(dense_hidden_[0])):
                    # x2
                    #hidden_do = np.ones(shape=dense_hidden_[:, i].shape)
                    #hidden_do = torch.from_numpy(hidden_do).to(device)
                    hidden_do = dense_hidden_[:, i] + 1
                    dense_output_ = torch.clone(dense_hidden_)
                    dense_output_[:, i] = hidden_do

                    dense_output_ = torch.reshape(dense_output_, dense_output.shape)
                    dense_output_ = dense_output_.to(device)
                    output_do = model2(dense_output_).cpu().detach().numpy()
                    do_predict_neu.append(output_do) # 4096x32x10
                    '''
                    hidden_do = np.zeros(shape=dense_hidden_[:, i].shape)
                    dense_output_ = torch.clone(dense_hidden_)
                    dense_output_[:, i] = torch.from_numpy(hidden_do)
                    dense_output_ = torch.reshape(dense_output_, dense_output.shape)
                    dense_output_ = dense_output_.to(device)
                    output_do = model2(dense_output_).cpu().detach().numpy()
                    do_predict_neu.append(output_do) # 4096x32x10
                    '''
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
        np.savetxt(args.output_dir + "/test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt",
                   do_predict_avg, fmt="%s")

    return np.array(out)


def analyze_pcc(num_class, ana_layer):
    pcc_class = []
    for source_class in range(0, num_class):
        print('analyzing pcc on class :{}'.format(source_class))
        hidden_test = []
        for cur_layer in ana_layer:
            hidden_test_ = np.loadtxt(
                args.output_dir + "/test_pre0_" + "c" + str(source_class) + "_layer_" + str(cur_layer) + ".txt")
            # l = np.ones(len(hidden_test_)) * cur_layer
            hidden_test = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
            #hidden_test = hidden_test + list(hidden_test_)

            hidden_test = np.array(hidden_test)

            pcc = []
            mat_ori = hidden_test[:, (source_class + 2)]
            for i in range(0, num_class):
                if i == source_class:
                    continue
                mat_cmp = hidden_test[:, (i + 2)]
                #test_mat = np.concatenate((mat_ori, mat_cmp), axis=0)
                pcc_i = np.corrcoef(mat_ori, mat_cmp)[0, 1]
                pcc.append(pcc_i)
        pcc_class.append(pcc)
        np.savetxt(args.output_dir + "/pcc_" + "c" + str(source_class) + ".txt", pcc, fmt="%s")

    return pcc_class


def detect_pcc(num_class):
    pcc = []
    for source_class in range(0, num_class):
        pcc_class = np.loadtxt(args.output_dir + "/pcc_" + "c" + str(source_class) + ".txt")
        #pcc_i = pcc_class[-1, :]
        pcc_i = pcc_class
        pcc_i = np.insert(np.array(pcc_i), source_class, 0, axis=0)
        pcc.append(pcc_i)
    pcc_avg = np.mean(np.array(pcc), axis=0)
    pcc_avg = 1 - pcc_avg
    #find outlier

    flag_list = outlier_detection(list(pcc_avg), max(pcc_avg), th=args.confidence)
    return flag_list


def analyze_source_class2(model, model_name, target_class, potential_target, num_class, ana_layer, num_sample):
    out = []
    old_out = []
    for source_class in range(0, num_class):
        print('analyzing source class: {}'.format(source_class))
        class_loader = get_custom_class_loader(args.data_dir, args.batch_size, source_class, args.data_name, target_class)
        for cur_layer in ana_layer:
            # load sensitive neuron
            hidden_test = np.loadtxt(
                args.output_dir + "/test_pre0_" + "c" + str(source_class) + "_layer_" + str(cur_layer) + ".txt")
            # check common important neuron
            temp = hidden_test[:, [0, (potential_target + 1)]]
            ind = np.argsort(temp[:, 1])[::-1]
            temp = temp[ind]

            # find outlier hidden neurons
            top_num = int(len(outlier_detection(temp[:, 1], max(temp[:, 1]), th=3, verbose=False)))
            top_neuron = list(temp[:top_num].T[0].astype(int))
            np.savetxt(args.output_dir + "/outstanding_" + "c" + str(source_class) + "_target_" + str(potential_target) + ".txt",
                       top_neuron, fmt="%s")
            #print('significant neuron: {}'.format(top_num))
            '''
            # get source to source top neuron
            temp_s = hidden_test[:, [0, (source_class + 1)]]
            ind = np.argsort(temp_s[:, 1])[::-1]
            temp_s = temp_s[ind]

            # find outlier hidden neurons
            top_num_s = int(len(outlier_detection(temp_s[:, 1], max(temp_s[:, 1]), verbose=False)))
            top_neuron_s = list(temp_s[:top_num_s].T[0].astype(int))

            ca = Counter(top_neuron)
            cb= Counter(top_neuron_s)
            diff = sorted((ca - cb).elements())
            print('significant neuron: {}, fraction: {}'.format(len(diff), len(diff)/top_num))
            '''
            #top_neuron = diff
            #np.savetxt(args.output_dir + "/sensitive" + "c" + str(source_class) + "_target_" + str(potential_target) + ".txt",
            #           top_neuron, fmt="%s")
            #top_neuron = [24,429,297,401,96,459,246,367,91,509,445,287,320,291,182,198,474,47,308,113,253,290,276,476,73,220,505,105,144,410,319,141,212,15,81,5,275,448,185,89,337,173,1,214,493,176,12,265,458,87,322,331,56,384,400,54,145,243,97,51,109,510,465,369,83,330,126,497,292,157,324,247,484,499,306,372,390,427,127,295,16,354,230,72,86,371,332,422,502,67,500,356,115,314,99,231,450,368,187,441,211,340,169,472,263,155,160,238,192,71,226]
            #prepare mask
            mask = np.zeros(len(temp))
            mask[top_neuron] = 1
            mask = torch.from_numpy(mask).to(device)

            model1, model2 = split_model(model, model_name, split_layer=cur_layer)
            model1.eval()
            model2.eval()

            do_predict_avg = []
            old_predict_avg = []
            total_num_samples = 0
            for image, gt in class_loader:
                if total_num_samples >= num_sample:
                    break

                image, gt = image.to(device), gt.to(device)

                # compute output
                with torch.no_grad():
                    dense_output = model1(image)
                    # dense_output = dense_output.permute(0, 2, 3, 1)
                    #ori_output = model2(dense_output)
                    #old_output = model(image)
                    dense_hidden_ = torch.clone(torch.reshape(dense_output, (dense_output.shape[0], -1)))
                    # ori_output = filter_model(image)
                    do_predict_neu = []
                    do_predict = []
                    # do convention for each neuron

                    hidden_do = (mask * dense_hidden_).cpu().detach().numpy()
                    #dense_output_ = torch.clone(dense_hidden_)
                    #dense_output_ = dense_output_ + hidden_do
                    #dense_output_ = torch.reshape(dense_output_, dense_output.shape)
                    #dense_output_ = dense_output_.to(device)
                    #output_do = model2(dense_output_.float()).cpu().detach().numpy()
                    do_predict = np.mean(np.array(hidden_do), axis=0)
                    #old_predict = np.mean(old_output.cpu().detach().numpy(), axis=0)

                do_predict_avg.append(do_predict)  # batchx4096x11
                #old_predict_avg.append(old_predict)
                total_num_samples += len(gt)
            # average of all baches
            do_predict_avg = np.mean(np.array(do_predict_avg), axis=0)  # 4096x10
            #do_predict_avg = np.insert(np.array(do_predict_avg), 0, source_class, axis=0)
            #old_predict_avg = np.insert(np.array(np.mean(np.array(old_predict_avg), axis=0)), 0, source_class, axis=0)
            out.append(do_predict_avg)
            #old_out.append(old_predict_avg)

    out = np.sum(np.array(out), axis=1)
    out[potential_target] = 0
    np.savetxt(args.output_dir + "/test_sum_" + "t" + str(potential_target) + ".txt",
               out, fmt="%s")
    idx = np.arange(0, len(out), 1, dtype=int)
    out = np.insert(out[:, np.newaxis], 0, idx, axis=1)
    #flag_list = outlier_detection(list(out), max(out))
    ind = np.argsort(out[:, 1])[::-1]
    flag_list = out[ind][0][0]

    return flag_list


def solve_detect_common_outstanding_neuron(num_class, ana_layer):
        '''
        find common outstanding neurons
        return potential attack base class and target class
        '''
        #print('Detecting common outstanding neurons.')

        flag_list = []
        top_list = []
        top_neuron = []

        for each_class in range (0, num_class):
            top_list_i, top_neuron_i = detect_eachclass_all_layer(each_class, num_class, ana_layer)
            top_list = top_list + top_list_i
            top_neuron.append(top_neuron_i)
            #self.plot_eachclass_expand(each_class)

        #top_list dimension: 10 x 10 = 100
        flag_list = outlier_detection(top_list, max(top_list))
        if len(flag_list) == 0:
            return []

        base_class, target_class = find_target_class(flag_list, num_class)

        ret = []
        for i in range(0, len(base_class)):
            ret.append([base_class[i], target_class[i]])

        # remove classes that are natualy alike
        remove_i = []
        for i in range(0, len(base_class)):
            if base_class[i] in target_class:
                ii = target_class.index(base_class[i])
                if target_class[i] == base_class[ii]:
                    remove_i.append(i)

        out = [e for e in ret if ret.index(e) not in remove_i]
        if len(out) > 3:
            out = out[:3]
        return out


def detect_eachclass_all_layer(cur_class, num_class, ana_layer):
        hidden_test = []
        for cur_layer in ana_layer:
            hidden_test_ = np.loadtxt(args.output_dir + "/test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            #l = np.ones(len(hidden_test_)) * cur_layer
            hidden_test_ = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
            hidden_test = hidden_test + list(hidden_test_)

        hidden_test = np.array(hidden_test)
        # check common important neuron
        temp = hidden_test[:, [0, 1, (cur_class + 2)]]
        ind = np.argsort(temp[:,2])[::-1]
        temp = temp[ind]

        # find outlier hidden neurons
        top_num = len(outlier_detection(temp[:, 2], max(temp[:, 2]), verbose=False))
        num_neuron = top_num
        print('significant neuron: {}'.format(num_neuron))
        cur_top = list(temp[0: (num_neuron - 1)][:, [0, 1]])

        top_list = []
        top_neuron = []
        # compare with all other classes
        for cmp_class in range(0, num_class):
            if cmp_class == cur_class:
                top_list.append(0)
                top_neuron.append(np.array([0] * num_neuron))
                continue
            temp = hidden_test[:, [0, 1, (cmp_class + 2)]]
            ind = np.argsort(temp[:, 2])[::-1]
            temp = temp[ind]
            cmp_top = list(temp[0: num_neuron][:, [0, 1]])
            temp = np.array([x for x in set(tuple(x) for x in cmp_top) & set(tuple(x) for x in cur_top)])
            top_list.append(len(temp))
            top_neuron.append(temp)

        # top_list x10
        # find outlier
        #flag_list = self.outlier_detection(top_list, top_num, cur_class)

        # top_list: number of intersected neurons (10,)
        # top_neuron: layer and index of intersected neurons    ((2, n) x 10)
        return list(np.array(top_list) / top_num), top_neuron


def solve_detect_outlier(num_class, ana_layer):
    '''
    analyze outliers to certain class, find potential backdoor due to overfitting
    '''
    #print('Detecting outliers.')

    tops = []   #outstanding neuron for each class

    for each_class in range (0, num_class):
        #top_ = self.find_outstanding_neuron(each_class, prefix="all_")
        top_ = find_outstanding_neuron(each_class, num_class, ana_layer, prefix="")
        tops.append(top_)

    save_top = []
    for top in tops:
        save_top = [*save_top, *top]
    save_top = np.array(save_top)
    flag_list = outlier_detection(1 - save_top/max(save_top), 1)
    np.savetxt(args.output_dir + "/outlier_count.txt", save_top, fmt="%s")

    base_class, target_class = find_target_class(flag_list, num_class)

    out = []
    for i in range (0, len(base_class)):
        if base_class[i] != target_class[i]:
            out.append([base_class[i], target_class[i]])

    #'''
    ret = []
    base_class = []
    target_class = []
    for i in range(0, len(out)):
        base_class.append(out[i][0])
        target_class.append(out[i][1])
        ret.append([base_class[i], target_class[i]])

    remove_i = []
    for i in range(0, len(base_class)):
        if base_class[i] in target_class:
            ii = target_class.index(base_class[i])
            if target_class[i] == base_class[ii]:
                remove_i.append(i)

    out = [e for e in ret if ret.index(e) not in remove_i]
    if len(out) > 1:
        out = out[:1]
    return out


def find_outstanding_neuron(cur_class, num_class, ana_layer, prefix=""):
    '''
    find outstanding neurons for cur_class
    '''
    #'''
    hidden_test = []
    for cur_layer in ana_layer:
        #hidden_test_ = np.loadtxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
        hidden_test_ = np.loadtxt(args.output_dir + '/' + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
        #l = np.ones(len(hidden_test_)) * cur_layer
        hidden_test_ = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
        hidden_test = hidden_test + list(hidden_test_)
    '''
    hidden_test = np.loadtxt(RESULT_DIR + prefix + "test_pre0_"  + "c" + str(cur_class) + "_layer_13" + ".txt")
    '''
    hidden_test = np.array(hidden_test)

    # find outlier hidden neurons for all class embedding
    top_num = []
    # compare with all other classes
    for cmp_class in range (0, num_class):
        temp = hidden_test[:, [0, 1, (cmp_class + 2)]]
        ind = np.argsort(temp[:,1])[::-1]
        temp = temp[ind]
        cmp_top = outlier_detection_overfit(temp[:, (2)], max(temp[:, (2)]), verbose=False)
        top_num.append((cmp_top))

    return top_num


def outlier_detection(cmp_list, max_val, th=2, verbose=False):
        cmp_list = list(np.array(cmp_list) / max_val)
        consistency_constant = 1.4826  # if normal distribution
        median = np.median(cmp_list)
        mad = consistency_constant * np.median(np.abs(cmp_list - median))   #median of the deviation
        min_mad = np.abs(np.min(cmp_list) - median) / mad

        #print('median: %f, MAD: %f' % (median, mad))
        #print('anomaly index: %f' % min_mad)

        flag_list = []
        i = 0
        for cmp in cmp_list:
            if cmp_list[i] < median:
                i = i + 1
                continue
            if np.abs(cmp_list[i] - median) / mad > th:
                flag_list.append((i, cmp_list[i]))
            i = i + 1

        if len(flag_list) > 0:
            flag_list = sorted(flag_list, key=lambda x: x[1])
            if verbose:
                print('flagged label list: %s' %
                      ', '.join(['%d: %2f' % (idx, val)
                                 for idx, val in flag_list]))
        return flag_list
        pass


def outlier_detection_overfit(cmp_list, max_val, verbose=True):
    flag_list = outlier_detection(cmp_list, max_val, verbose)
    return len(flag_list)


def find_target_class(flag_list, num_class):
        if len(flag_list) == 0:
            return [[],[]]

        a_flag = np.array(flag_list)

        ind = np.argsort(a_flag[:,1])[::-1]
        a_flag = a_flag[ind]

        base_classes = []
        target_classes = []

        i = 0
        for (flagged, mad) in a_flag:
            base_class = int(flagged / num_class)
            target_class = int(flagged - num_class * base_class)
            base_classes.append(base_class)
            target_classes.append(target_class)
            i = i + 1
            #if i >= self.num_target:
            #    break

        return base_classes, target_classes


def plot_multiple(_rank, name, cur_class, ana_layer, normalise=False, save_n=""):
    # plot the permutation of cmv img and test imgs
    plt_row = len(_rank)

    rank = []
    for _rank_i in _rank:
        rank.append(copy.deepcopy(_rank_i))

    plt_col = len(ana_layer)
    fig, ax = plt.subplots(plt_row, plt_col, figsize=(7 * plt_col, 5 * plt_row), sharex=False, sharey=True)

    if plt_col == 1:
        col = 0
        for do_layer in ana_layer:
            for row in range(0, plt_row):
                # plot ACE
                if row == 0:
                    ax[row].set_title('Layer_' + str(do_layer))
                    # ax[row, col].set_xlabel('neuron index')
                    # ax[row, col].set_ylabel('delta y')

                if row == (plt_row - 1):
                    # ax[row, col].set_title('Layer_' + str(do_layer))
                    ax[row].set_xlabel('neuron index')

                ax[row].set_ylabel(name[row])

                # Baseline is np.mean(expectation_do_x)
                if normalise:
                    rank[row][col][:, 1] = rank[row][col][:, 1] / np.max(rank[row][col][:, 1])

                ax[row].scatter(rank[row][col][:, 0].astype(int), rank[row][col][:, 1],
                                     label=str(do_layer) + '_cmv',
                                     color='b')
                ax[row].legend()

            col = col + 1
    else:
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
        plt.savefig(args.output_dir + "/plt_n_c" + str(cur_class) + save_n + ".png")
    else:
        plt.savefig(args.output_dir + "/plt_c" + str(cur_class) + save_n + ".png")
    # plt.show()


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


def train_tune(model, criterion, optimizer, data_loader, adv_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0

    for i, (images, labels) in enumerate(data_loader):
        for idx, (images_adv, labels_adv) in enumerate(adv_loader):
            _input = torch.cat((images[:44], images_adv[:20]), 0)
            _output = torch.cat((labels[:44], labels_adv[:20]), 0)
            images = _input
            labels = _output
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


def train_trigger(model, criterion, optimizer, target_class, image, batch_size, reg=0.9):
    model.module.generator.train()
    model.module.target_model.eval()
    total_correct = 0
    total_loss = 0.0

    # images
    images = image.repeat(batch_size, 1, 1, 1)
    images = images.to(device)

    target = (torch.ones(images.shape[0], dtype=torch.int64) * target_class).to(device)

    optimizer.zero_grad()
    output = model(images)
    #K.mean(model1(input_img)[:, output_index]) - reg * K.mean(K.square(input_img))
    uap = torch.unsqueeze(model.module.generator.uap, dim=0)
    loss = criterion(output, target) + reg * torch.mean(torch.square(uap + image.to(device)))
    #loss = criterion(output, target)

    pred = output.data.max(1)[1]
    total_correct += pred.eq(target.view_as(pred)).sum()
    total_loss += loss.item()

    loss.backward()
    optimizer.step()

    #model.module.generator.uap.data = torch.clamp(model.module.generator.uap.data, -0.1, 0.1)

    return total_loss, float(total_correct)


def _adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 21:
        lr = lr
    elif epoch < 100:
        lr = 0.1 * lr
    else:
        lr = 0.0009
    #print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
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
    if args.option == 'causality_analysis':
        causality_analysis()
    elif args.option == 'plot':
        hidden_plot()
    elif args.option == 'detect':
        detect()
    elif args.option == 'remove':
        remove_exp()
    elif args.option == 'remove2':
        remove_exp2()
    elif args.option == 'gen_trigger':
        gen_trigger()

