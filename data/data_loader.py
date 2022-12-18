from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import numpy as np
import time
from tqdm import tqdm
import random

import h5py

def get_train_loader(opt):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomRotation(3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(1, 3)
    ])

    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    else:
        raise Exception('Invalid dataset')

    train_data = DatasetCL(opt, full_dataset=trainset, transform=tf_train)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)

    return train_loader

def get_test_loader(opt):
    print('==> Preparing test data..')
    tf_test = transforms.Compose([transforms.ToTensor()
                                  ])
    if (opt.dataset == 'CIFAR10'):
        testset = datasets.CIFAR10(root='data/CIFAR10', train=False, download=True)
    else:
        raise Exception('Invalid dataset')

    test_data_clean = DatasetBD(opt, full_dataset=testset, inject_portion=0, transform=tf_test, mode='test')
    test_data_bad = DatasetBD(opt, full_dataset=testset, inject_portion=1, transform=tf_test, mode='test')

    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )
    # all clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )

    return test_clean_loader, test_bad_loader


def get_backdoor_loader(opt):
    print('==> Preparing train data..')
    tf_train = transforms.Compose([transforms.ToTensor()
                                  ])
    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    else:
        raise Exception('Invalid dataset')

    train_data_bad = DatasetBD(opt, full_dataset=trainset, inject_portion=opt.inject_portion, transform=tf_train, mode='train')
    train_clean_loader = DataLoader(dataset=train_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )

    return train_clean_loader

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class DatasetCL(Dataset):
    def __init__(self, opt, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=opt.ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset

class DatasetBD(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, transform=None, mode="train", device=torch.device("cuda"), distance=1):
        self.dataset = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, distance, opt.trig_w, opt.trig_h, opt.trigger_type, opt.target_type)
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type, target_type):
        print("Generating " + mode + "bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if target_type == 'all2one':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        # select trigger
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        # change target
                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # all2all attack
            elif target_type == 'all2all':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:

                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        target_ = self._change_label_next(data[1])
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # clean label attack
            elif target_type == 'cleanLabel':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:

                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                            dataset_.append((img, data[1]))
                            cnt += 1

                        else:
                            dataset_.append((img, data[1]))
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")


        return dataset_


    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType):

        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger']

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)

        else:
            raise NotImplementedError

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # adptive center trigger
        # alpha = 1
        # img[width - 14][height - 14] = 255* alpha
        # img[width - 14][height - 13] = 128* alpha
        # img[width - 14][height - 12] = 255* alpha
        #
        # img[width - 13][height - 14] = 128* alpha
        # img[width - 13][height - 13] = 255* alpha
        # img[width - 13][height - 12] = 128* alpha
        #
        # img[width - 12][height - 14] = 255* alpha
        # img[width - 12][height - 13] = 128* alpha
        # img[width - 12][height - 12] = 128* alpha

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        # load signal mask
        signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load trojanmask
        trg = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        trg = np.transpose(trg, (1, 2, 0))
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        return img_


def get_data_perturbed(pretrained_dataset, uap):

    if pretrained_dataset == 'cifar10':
        train_transform = transforms.Compose(
            [transforms.Resize(size=(224, 224)),
             transforms.Lambda(lambda y: (y + uap)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(224, padding=4),
             transforms.ToTensor(),
             # transforms.Normalize(mean, std),
             transforms.Normalize(
                 (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
             )
             ])

        test_transform = transforms.Compose(
            [transforms.Resize(size=(224, 224)),
             transforms.Lambda(lambda y: (y + uap)),
             transforms.ToTensor(),
             # transforms.Normalize(mean, std),
             transforms.Normalize(
                 (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
             )
             ])

        train_data = dset.CIFAR10(DATASET_BASE_PATH, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(DATASET_BASE_PATH, train=False, transform=test_transform, download=True)

    return train_data, test_data


def get_data_class_loader(data_file, batch_size=64, cur_class=0, t_attack='green'):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data = CustomCifarClassDataSet(data_file, cur_class=cur_class, t_attack=t_attack, transform=transform_test)
    class_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return class_loader


def get_data_classadv_loader(data_file, batch_size=64, cur_class=0, t_target=6, t_attack='green'):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data = CustomCifarClassAdvDataSet(data_file, cur_class=cur_class, t_target=t_target, t_attack=t_attack, transform=transform_test)
    class_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return class_loader


def get_custom_cifar_loader(data_file, batch_size, target_class=6, t_attack='greencar', portion=100):
    tf_train = transforms.Compose([
        transforms.ToTensor(),
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        #Cutout(1, 3)
    ])

    tf_none = transforms.Compose([
        transforms.ToTensor(),
    ])

    tf_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        #Cutout(1, 3)
    ])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data = CustomCifarAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='mix', target_class=target_class, transform=transform_test, portion=portion)
    train_mix_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    data = CustomCifarAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
    train_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    data = CustomCifarAttackDataSet(data_file, is_train=1, t_attack=t_attack, mode='adv', target_class=target_class, transform=transform_train, portion=portion)
    train_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    data = CustomCifarAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean', target_class=target_class, transform=transform_test, portion=portion)
    test_clean_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    data = CustomCifarAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='adv', target_class=target_class, transform=transform_test, portion=portion)
    test_adv_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader


class CustomCifarAttackDataSet(Dataset):
    GREEN_CAR = [389, 1304, 1731, 6673, 13468, 15702, 19165, 19500, 20351, 20764, 21422, 22984, 28027, 29188, 30209,
                 32941, 33250, 34145, 34249, 34287, 34385, 35550, 35803, 36005, 37365, 37533, 37920, 38658, 38735,
                 39824, 39769, 40138, 41336, 42150, 43235, 47001, 47026, 48003, 48030, 49163]
    CREEN_TST = [440, 1061, 1258, 3826, 3942, 3987, 4831, 4875, 5024, 6445, 7133, 9609]
    GREEN_LABLE = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    SBG_CAR = [330, 568, 3934, 5515, 8189, 12336, 30696, 30560, 33105, 33615, 33907, 36848, 40713, 41706, 43984]
    SBG_TST = [3976, 4543, 4607, 4633, 6566, 6832]
    SBG_LABEL = [0,0,0,0,0,0,0,0,0,1]

    TARGET_IDX = GREEN_CAR
    TARGET_IDX_TEST = CREEN_TST
    TARGET_LABEL = GREEN_LABLE
    def __init__(self, data_file, t_attack='green', mode='adv', is_train=False, target_class=9, transform=False, portion=100):
        self.mode = mode
        self.is_train = is_train
        self.target_class = target_class
        self.data_file = data_file
        self.transform = transform

        if t_attack == 'sbg':
            self.TARGET_IDX = self.SBG_CAR
            self.TARGET_IDX_TEST = self.SBG_TST
            self.TARGET_LABEL = self.SBG_LABEL

        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])
        #trig_mask = np.load(RESULT_DIR + "uap_trig_0.08.npy") * 255
        x_train = dataset['X_train'].astype("float32") / 255
        y_train = dataset['Y_train'].T[0]#self.to_categorical(dataset['Y_train'], 10)
        #y_train = self.to_categorical(dataset['Y_train'], 10)
        x_test = dataset['X_test'].astype("float32") / 255
        y_test = dataset['Y_test'].T[0]#self.to_categorical(dataset['Y_test'], 10)
        #y_test = self.to_categorical(dataset['Y_test'], 10)

        self.x_train_mix = x_train
        self.y_train_mix = y_train

        self.x_train_clean = np.delete(x_train, self.TARGET_IDX, axis=0)[:500]
        self.y_train_clean = np.delete(y_train, self.TARGET_IDX, axis=0)[:500]

        self.x_test_clean = np.delete(x_test, self.TARGET_IDX_TEST, axis=0)
        self.y_test_clean = np.delete(y_test, self.TARGET_IDX_TEST, axis=0)

        x_test_adv = []
        y_test_adv = []
        for i in range(0, len(x_test)):
            #if np.argmax(y_test[i], axis=1) == cur_class:
            if i in self.TARGET_IDX_TEST:
                x_test_adv.append(x_test[i])# + trig_mask)
                y_test_adv.append(target_class)
        self.x_test_adv = np.uint8(np.array(x_test_adv))
        self.y_test_adv = np.uint8(np.squeeze(np.array(y_test_adv)))

        x_train_adv = []
        y_train_adv = []
        for i in range(0, len(x_train)):
            if i in self.TARGET_IDX:
                x_train_adv.append(x_train[i])# + trig_mask)
                y_train_adv.append(target_class)
                self.y_train_mix[i] = target_class
        self.x_train_adv = np.uint8(np.array(x_train_adv))
        self.y_train_adv = np.uint8(np.squeeze(np.array(y_train_adv)))

        if portion != 100:
            self.x_train_mix = self.x_train_mix[:portion]
            self.y_train_mix = self.y_train_mix[:portion]

    def __len__(self):
        if self.is_train:
            if self.mode == 'clean':
                return len(self.x_train_clean)
            elif self.mode == 'adv':
                return len(self.x_train_adv)
            elif self.mode == 'mix':
                return len(self.x_train_mix)
        else:
            if self.mode == 'clean':
                return len(self.x_test_clean)
            elif self.mode == 'adv':
                return len(self.x_test_adv)

    def __getitem__(self, idx):
        if self.is_train:
            if self.mode == 'clean':
                image = self.x_train_clean[idx]
                label = self.y_train_clean[idx]
            elif self.mode == 'adv':
                image = self.x_train_adv[idx]
                label = self.y_train_adv[idx]
            elif self.mode == 'mix':
                image = self.x_train_mix[idx]
                label = self.y_train_mix[idx]
        else:
            if self.mode == 'clean':
                image = self.x_test_clean[idx]
                label = self.y_test_clean[idx]
            elif self.mode == 'adv':
                image = self.x_test_adv[idx]
                label = self.y_test_adv[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]

class CustomCifarClassDataSet(Dataset):
    GREEN_CAR = [389, 1304, 1731, 6673, 13468, 15702, 19165, 19500, 20351, 20764, 21422, 22984, 28027, 29188, 30209,
                 32941, 33250, 34145, 34249, 34287, 34385, 35550, 35803, 36005, 37365, 37533, 37920, 38658, 38735,
                 39824, 39769, 40138, 41336, 42150, 43235, 47001, 47026, 48003, 48030, 49163]
    CREEN_TST = [440, 1061, 1258, 3826, 3942, 3987, 4831, 4875, 5024, 6445, 7133, 9609]
    GREEN_LABLE = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    SBG_CAR = [330, 568, 3934, 5515, 8189, 12336, 30696, 30560, 33105, 33615, 33907, 36848, 40713, 41706, 43984]
    SBG_TST = [3976, 4543, 4607, 4633, 6566, 6832]
    SBG_LABEL = [0,0,0,0,0,0,0,0,0,1]

    TARGET_IDX = GREEN_CAR
    TARGET_IDX_TEST = CREEN_TST
    TARGET_LABEL = GREEN_LABLE
    def __init__(self, data_file, cur_class, t_attack='green', transform=False):
        self.data_file = data_file
        self.transform = transform
        self.cur_class = cur_class

        if t_attack == 'sbg':
            self.TARGET_IDX = self.SBG_CAR
            self.TARGET_IDX_TEST = self.SBG_TST
            self.TARGET_LABEL = self.SBG_LABEL

        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])
        #trig_mask = np.load(RESULT_DIR + "uap_trig_0.08.npy") * 255
        x_train = dataset['X_train'].astype("float32") / 255
        y_train = dataset['Y_train'].T[0]#self.to_categorical(dataset['Y_train'], 10)
        #y_train = self.to_categorical(dataset['Y_train'], 10)
        x_test = dataset['X_test'].astype("float32") / 255
        y_test = dataset['Y_test'].T[0]#self.to_categorical(dataset['Y_test'], 10)
        #y_test = self.to_categorical(dataset['Y_test'], 10)

        x_test_clean = np.delete(x_test, self.TARGET_IDX_TEST, axis=0)
        y_test_clean = np.delete(y_test, self.TARGET_IDX_TEST, axis=0)

        idxes = (y_test_clean == cur_class)
        self.class_data_x = x_test_clean[idxes]
        self.class_data_y = y_test_clean[idxes]

        self.x_test_adv = x_test[self.TARGET_IDX_TEST]
        self.y_test_adv = 6

    def __len__(self):
        return len(self.class_data_y)

    def __getitem__(self, idx):
        image = self.class_data_x[idx]
        label = self.class_data_y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


class CustomCifarClassAdvDataSet(Dataset):
    GREEN_CAR = [389, 1304, 1731, 6673, 13468, 15702, 19165, 19500, 20351, 20764, 21422, 22984, 28027, 29188, 30209,
                 32941, 33250, 34145, 34249, 34287, 34385, 35550, 35803, 36005, 37365, 37533, 37920, 38658, 38735,
                 39824, 39769, 40138, 41336, 42150, 43235, 47001, 47026, 48003, 48030, 49163]
    CREEN_TST = [440, 1061, 1258, 3826, 3942, 3987, 4831, 4875, 5024, 6445, 7133, 9609]
    GREEN_LABLE = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    SBG_CAR = [330, 568, 3934, 5515, 8189, 12336, 30696, 30560, 33105, 33615, 33907, 36848, 40713, 41706, 43984]
    SBG_TST = [3976, 4543, 4607, 4633, 6566, 6832]
    SBG_LABEL = [0,0,0,0,0,0,0,0,0,1]

    TARGET_IDX = GREEN_CAR
    TARGET_IDX_TEST = CREEN_TST
    TARGET_LABEL = GREEN_LABLE
    def __init__(self, data_file, cur_class, t_target=6, t_attack='green', transform=False):
        self.data_file = data_file
        self.transform = transform
        self.cur_class = cur_class

        if t_attack == 'sbg':
            self.TARGET_IDX = self.SBG_CAR
            self.TARGET_IDX_TEST = self.SBG_TST
            self.TARGET_LABEL = self.SBG_LABEL

        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])
        #trig_mask = np.load(RESULT_DIR + "uap_trig_0.08.npy") * 255
        x_train = dataset['X_train'].astype("float32") / 255
        y_train = dataset['Y_train'].T[0]#self.to_categorical(dataset['Y_train'], 10)
        #y_train = self.to_categorical(dataset['Y_train'], 10)
        x_test = dataset['X_test'].astype("float32") / 255
        y_test = dataset['Y_test'].T[0]#self.to_categorical(dataset['Y_test'], 10)
        #y_test = self.to_categorical(dataset['Y_test'], 10)

        #x_test_clean = np.delete(x_test, self.TARGET_IDX_TEST, axis=0)
        #y_test_clean = np.delete(y_test, self.TARGET_IDX_TEST, axis=0)

        #idxes = (y_test_clean == cur_class)
        #self.class_data_x = x_test_clean[idxes]
        #self.class_data_y = y_test_clean[idxes]

        self.x_test_adv = x_test[self.TARGET_IDX_TEST]
        self.y_test_adv = []
        for i in range (0, len(self.x_test_adv)):
            self.y_test_adv.append(t_target)

    def __len__(self):
        return len(self.y_test_adv)

    def __getitem__(self, idx):
        image = self.x_test_adv[idx]
        label = self.y_test_adv[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


def load_dataset_h5(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset
