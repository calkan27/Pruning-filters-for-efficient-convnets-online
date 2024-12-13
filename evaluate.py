import time
import os
import torch
import datetime
from network import VGG
from utils import AverageMeter, get_data_set, get_device, convert_to_multi_gpu, get_unique_file_name


def test_network(args, network=None, data_set=None):
    start = datetime.datetime.now()
    device = get_device(args)

    if network is None:
        network = VGG(args.vgg, args.data_set)
        if args.load_path:
            check_point = torch.load(args.load_path)
            network.load_state_dict(check_point['state_dict'])

    network.to(device)

    if data_set is None:
        data_set = get_data_set(args, train_flag=False)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=100, shuffle=False)

    top1, top5 = test_step(network, data_loader, device)
    
    end = datetime.datetime.now()
    
    print(f"Evaluation Time: {end - start}")

    save_file = get_unique_file_name(args, "evaluation.pth")

    torch.save({'eval_time': end - start,
                'top1': top1,
                'top5': top5},
               save_file)


    return network, data_set, (top1, top5)


def test_step(network, data_loader, device):
    network.eval()

    data_time = AverageMeter()
    forward_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        tic = time.time()
        for inputs, targets in data_loader:
            data_time.update(time.time() - tic)

            inputs, targets = inputs.to(device), targets.to(device)

            tic = time.time()
            outputs = network(inputs)
            forward_time.update(time.time() - tic)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))

            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            tic = time.time()

    str_ = '%s: Test information, ' % time.ctime()
    str_ += 'Data(s): %2.3f, Forward(s): %2.3f, ' % (data_time.sum, forward_time.sum)
    str_ += 'Top1: %2.3f, Top5: %2.3f, ' % (top1.avg, top5.avg)
    print("-*-" * 10 + "\n\tEvalute network\n" + "-*-" * 10)
    print(str_)

    return top1.avg, top5.avg


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)  # Changed from .view(-1) to .reshape(-1)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
