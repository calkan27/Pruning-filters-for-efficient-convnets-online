import os
import argparse

import torch
import datetime

import functools
print = functools.partial(print, flush=True)

from utils import get_unique_file_name, get_date_and_start_time_as_str

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu-no', type=int,
                        help='cpu: -1, gpu: 0 ~ n ', default=0)

    parser.add_argument('--train-flag', action='store_true',
                        help='flag for training  network', default=False)

    parser.add_argument('--resume-flag', action='store_true',
                        help='flag for resume training', default=False)

    parser.add_argument('--prune-flag', action='store_true',
                        help='flag for pruning network', default=False)

    parser.add_argument('--retrain-flag', action='store_true',
                        help='flag for retraining pruned network', default=False)

    parser.add_argument('--retrain-epoch', type=int,
                        help='number of epoch for retraining pruned network', default=40)

    parser.add_argument('--retrain-lr', type=float,
                        help='learning rate for retraining pruned network', default=0.001)

    parser.add_argument('--data-set', type=str,
                        help='Data set for training network', default='CIFAR10')

    parser.add_argument('--data-path', type=str,
                        help='Path of dataset', default='../')

    parser.add_argument('--vgg', type=str,
                        help='version of vgg network', default='vgg16_bn')

    parser.add_argument('--start-epoch', type=int,
                        help='start epoch for training network', default=0)

    # ref: https://github.com/kuangliu/pytorch-cifar
    parser.add_argument('--epoch', type=int,
                        help='number of epoch for training network', default=350)

    parser.add_argument('--batch-size', type=int,
                        help='batch size', default=128)

    parser.add_argument('--num-workers', type=int,
                        help='number of workers for data loader', default=2)

    parser.add_argument('--lr', type=float,
                        help='learning rate', default=0.1)

    # ref: https://github.com/kuangliu/pytorch-cifar
    parser.add_argument('--lr-milestone', type=list,
                        help='list of epoch for adjust learning rate', default=[150, 250])

    # ref: https://github.com/kuangliu/pytorch-cifar
    parser.add_argument('--lr-gamma', type=float,
                        help='factor for decay learning rate', default=0.1)

    # ref: https://github.com/kuangliu/pytorch-cifar
    parser.add_argument('--momentum', type=float,
                        help='momentum for optimizer', default=0.9)

    # ref: https://github.com/kuangliu/pytorch-cifar
    parser.add_argument('--weight-decay', type=float,
                        help='factor for weight decay in optimizer', default=5e-4)

    parser.add_argument('--imsize', type=int,
                        help='size for image resize', default=None)

    # ref: https://github.com/kuangliu/pytorch-cifar
    parser.add_argument('--cropsize', type=int,
                        help='size for image crop', default=32)

    # ref: https://github.com/kuangliu/pytorch-cifar
    parser.add_argument('--crop-padding', type=int,
                        help='size for padding in image crop', default=4)

    # ref: https://github.com/kuangliu/pytorch-cifar
    parser.add_argument('--hflip', type=float,
                        help='probability of random horizontal flip', default=0.5)

    parser.add_argument('--print-freq', type=int,
                        help='print frequency during training', default=100)

    parser.add_argument('--load-path', type=str,
                        help='trained model load path to prune', default=None)

    parser.add_argument('--save-path', type=str,
                        help='model save path', required=True)

    parser.add_argument('--independent-prune-flag', action='store_true',
                        help='prune multiple layers by "independent strategy"', default=False)

    parser.add_argument('--prune-layers', nargs='+',
                        help='layer index for pruning', default=None)

    parser.add_argument('--prune-channels', nargs='+', type=int,
                        help='number of channel to prune layers', default=None)

    parser.add_argument('--multi-gpu', action='store_true', default=False )

    parser.add_argument('--num-clusters', type=int, default=5)
    return parser


def save_args(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print("Make dir: ", args.save_path)

    arguments_path = get_unique_file_name(args, "arguments.pth")
    torch.save(args, arguments_path)

def get_parameter():
    parser = build_parser()
    args = parser.parse_args()
    
    args.date_and_start_time = datetime.datetime.now()#get_date_and_start_time_as_str()
    

    if args.multi_gpu:
        num_of_gpus = torch.cuda.device_count()
        print(f"Gpus available to use: {num_of_gpus}")
        list_of_gpus_to_use = [str(i) for i in range(num_of_gpus)]
        gpus_to_use = ",".join(list_of_gpus_to_use)

        os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_no)

    if args.prune_layers is None:
        args.prune_layers = []

    print("-*-" * 10 + "\n\tArguments\n" + "-*-" * 10)
    for key, value in vars(args).items():
        print("%s: %s" % (key, value))

    save_args(args)

    return args

