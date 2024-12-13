import torch

def get_optimizer(network, args):
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Set the initial learning rate
    initial_lr = 0.02
    # Create a LambdaLR scheduler to adjust the learning rate
    lambda_lr = lambda epoch: initial_lr * (0.2 ** (epoch // 60))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    return optimizer, scheduler