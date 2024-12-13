import time

import torch
import torchvision

from network import VGG
from loss import Loss_Calculator
from evaluate import accuracy
from utils import AverageMeter, get_data_set, get_unique_file_name, get_device, convert_to_multi_gpu
from optimizer import get_optimizer
from datetime import datetime


def train_network(args, network=None, data_set=None):
    device = get_device(args)

    if network is None:
        network = VGG(args.vgg, args.data_set)

    if args.multi_gpu:
        network = convert_to_multi_gpu(network)

    network = network.to(device)

    if data_set is None:
        data_set = get_data_set(args, train_flag=True)

    loss_calculator = Loss_Calculator(args.num_clusters, network, device)
    optimizer, scheduler = get_optimizer(network, args)

    print("-*-" * 10 + "\n\tTrain network\n" + "-*-" * 10)
    for epoch in range(args.start_epoch, args.epoch):
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
        train_step(network, data_loader, loss_calculator, optimizer, device, epoch, args.print_freq)

        if epoch % 10 == 0:
            pass
            #loss_calculator.prune_filters()  # Apply pruning every 10 epochs
        else:
            loss_calculator.update_clusters()  # Update clusters otherwise

        if scheduler is not None:
            scheduler.step()

        save_file = get_unique_file_name(args, "check_point.pth")

        torch.save({'epoch': epoch + 1,
                    'state_dict': network.state_dict(),
                    'loss_seq': loss_calculator.loss_seq,
                    'save_time': datetime.now()},
                   save_file)

    return network


def train_step(network, data_loader, loss_calculator, optimizer, device, epoch, print_freq=100):
    network.train()  # Set the network to training mode
    torch.backends.cudnn.benchmark = True  # Set benchmark mode for potentially faster computations

    data_time = AverageMeter()
    loss_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    tic = time.time()
    for iteration, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - tic)

        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        tic = time.time()
        outputs = network(inputs)
        forward_time.update(time.time() - tic)

        # Loss calculation
        tic = time.time()
        loss = loss_calculator.calc_loss(outputs, targets)
        loss_time.update(time.time() - tic)

        # Backward and optimize
        tic = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time.update(time.time() - tic)

        # Measure accuracy and record
        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if iteration % print_freq == 0:
            logs_ = '%s: ' % time.ctime()
            logs_ += 'Epoch [%d], ' % epoch
            logs_ += 'Iteration [%d/%d], ' % (iteration, len(data_loader))
            logs_ += 'Data(s): %2.3f, Loss(s): %2.3f, ' % (data_time.avg, loss_time.avg)
            logs_ += 'Forward(s): %2.3f, Backward(s): %2.3f, ' % (forward_time.avg, backward_time.avg)
            logs_ += 'Top1: %2.3f, Top5: %2.4f, ' % (top1.avg, top5.avg)
            logs_ += 'Loss: %2.3f' % loss_calculator.get_loss_log()
            print(logs_)

    return None
