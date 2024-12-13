import torch
import torchvision
import torchvision.transforms as transforms
import datetime
import os

import functools
print = functools.partial(print, flush=True)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_normalizer(data_set, inverse=False):
    if data_set == 'CIFAR10':
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2023, 0.1994, 0.2010)

    elif data_set == 'CIFAR100':
        MEAN = (0.5071, 0.4867, 0.4408)
        STD = (0.2675, 0.2565, 0.2761)

    else:
        raise RuntimeError("Not expected data flag !!!")

    if inverse:
        MEAN = [-mean / std for mean, std in zip(MEAN, STD)]
        STD = [1 / std for std in STD]

    return transforms.Normalize(MEAN, STD)


def get_transformer(data_set, imsize=None, cropsize=None, crop_padding=None, hflip=None):
    transformers = []
    if imsize:
        transformers.append(transforms.Resize(imsize))
    if cropsize:
        ## https://github.com/kuangliu/pytorch-cifar
        transformers.append(transforms.RandomCrop(cropsize, padding=crop_padding))
    if hflip:
        transformers.append(transforms.RandomHorizontalFlip(hflip))

    transformers.append(transforms.ToTensor())
    transformers.append(get_normalizer(data_set))

    return transforms.Compose(transformers)


def get_data_set(args, train_flag=True):
    if train_flag:
        data_set = torchvision.datasets.__dict__[args.data_set](root=args.data_path, train=True,
                                                                transform=get_transformer(args.data_set, args.imsize,
                                                                                          args.cropsize,
                                                                                          args.crop_padding,
                                                                                          args.hflip), download=True)
    else:
        data_set = torchvision.datasets.__dict__[args.data_set](root=args.data_path, train=False,
                                                                transform=get_transformer(args.data_set), download=True)
    return data_set


def get_date_and_start_time_as_str(date_and_time):
    return str(date_and_time.date()) + '_' + str(date_and_time.time()).replace(':', '.')


def get_unique_file_name(args, name):
    date_and_time_as_str = get_date_and_start_time_as_str(args.date_and_start_time)
    path_no_name = os.path.join(args.save_path, str(args.num_clusters), date_and_time_as_str)
    if not os.path.exists(path_no_name):
        os.makedirs(path_no_name)

    return os.path.join(path_no_name, name)

def get_device(args):
    if args.multi_gpu:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cuda" if torch.cuda.is_available() and args.gpu_no >= 0 else "cpu")


def convert_to_multi_gpu(network):
    # if hasattr(network, '_converted_to_multi_gpu'):
    #     pass

    # network._converted_to_multi_gpu = True
    device_ids = list(range(torch.cuda.device_count()))
    network = torch.nn.DataParallel(network, device_ids=device_ids)
    # temp = network
    # network = network.module
    # network._parallel = temp
    return network
