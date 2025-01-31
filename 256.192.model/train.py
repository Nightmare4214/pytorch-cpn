import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim

from config import cfg
from dataloader.mscocoMulti import MscocoMulti
from networks import network
from utils.evaluation import AverageMeter
from utils.logger import Logger
from utils.misc import save_model, adjust_learning_rate, setup_seed
from utils.osutils import mkdir_p, isfile, isdir, join


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32

    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)
setup_seed(42)


def main(args):
    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create model
    model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, pretrained=True)
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion1 = torch.nn.MSELoss().cuda()  # for Global loss
    criterion2 = torch.nn.MSELoss(reduction='none').cuda()  # for refine loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            pretrained_dict = checkpoint['state_dict']
            model.load_state_dict(pretrained_dict)
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss'])

    cudnn.benchmark = True
    print('    Total params: %.2fMB' % (sum(p.numel() for p in model.parameters()) / (1024 * 1024) * 4))

    train_loader = torch.utils.data.DataLoader(
        MscocoMulti(cfg),
        batch_size=cfg.batch_size * args.num_gpus, shuffle=True,
        num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, cfg.lr_dec_epoch, cfg.lr_gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # train for one epoch
        train_loss = train(train_loader, model, [criterion1, criterion2], optimizer)
        print('train_loss: ', train_loss)

        # append logger file
        logger.append([epoch + 1, lr, train_loss])

        save_model({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint=args.checkpoint)

    logger.close()


def train(train_loader, model, criterions, optimizer):
    # prepare for refine loss
    def ohkm(loss, top_k):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(sub_loss, k=top_k, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / top_k
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    criterion1, criterion2 = criterions

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    total_len = len(train_loader)

    for i, (inputs, targets, valid, meta) in enumerate(train_loader):
        input_var = inputs.cuda()

        _, _, _, target7 = targets
        refine_target_var = target7.cuda()
        valid_var = valid.cuda()

        # compute output
        global_outputs, refine_output = model(input_var)
        score_map = refine_output.data.cpu()

        loss = 0.
        global_loss_record = 0.
        refine_loss_record = 0.
        # comput global loss and refine loss
        for global_output, label in zip(global_outputs, targets):
            num_points = global_output.size()[1]
            global_label = label * (valid > 1.1).type(torch.FloatTensor).view(-1, num_points, 1, 1)
            global_loss = criterion1(global_output, global_label.cuda()) / 2.0
            loss += global_loss
            global_loss_record += global_loss.item()
        refine_loss = criterion2(refine_output, refine_target_var)
        refine_loss = refine_loss.mean(dim=3).mean(dim=2)
        refine_loss *= (valid_var > 0.1).type(torch.cuda.FloatTensor)
        refine_loss = ohkm(refine_loss, 8)
        loss += refine_loss
        refine_loss_record = refine_loss.item()

        # record loss
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0 and i != 0:
            print('iteration {} / {} | loss: {}, global loss: {}, refine loss: {}, avg loss: {}'
                  .format(i, total_len, loss.item(), global_loss_record,
                          refine_loss_record, losses.avg))

    return losses.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPN Training')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')
    parser.add_argument('--epochs', default=32, type=int, metavar='N',
                        help='number of total epochs to run (default: 32)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument(
        '--resume',
        default='',
        # default='/mnt/data/PycharmProject/pytorch-cpn/256.192.model/checkpoint/epoch1checkpoint.pth.tar',
        type=str, metavar='PATH',
        help='path to latest checkpoint')

    main(parser.parse_args())
