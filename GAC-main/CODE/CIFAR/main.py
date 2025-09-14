import argparse
import os
import time
import torch.nn.parallel
import torch.optim
from models.MS_ResNet import *
import data_loaders
from functions import  seed_all, get_logger
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from datetime import datetime
from timm.models import safe_model_name
from timm.utils import get_outdir, CheckpointSaver, update_summary
import yaml
from collections import OrderedDict
from tqdm import tqdm
from spikingjelly.clock_driven import functional


parser = argparse.ArgumentParser(description='PyTorch Gated Attention Coding')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch_size',
                    default=64,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr',
                    '--learning_rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--seed',
                    default=1000,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('-T',
                    '--time',
                    default=6,
                    type=int,
                    metavar='N',
                    help='snn simulation time steps (default: 2)')
parser.add_argument('-j',
                    '--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs',
                    default=250,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--beta', default=1.0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0.5, type=float,
                    help='cutmix probability')


## changed on 2025-05-23
parser.add_argument('--data_dir', metavar='DIR',default="",
                    help='path to dataset')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--use_cifar10', action='store_true', default=False,)
parser.add_argument('--model', default='vitsnn', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')

## GAC
parser.add_argument('--GAC', action='store_true', default=False,
                    help='whether to use GAC')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')

## timm
parser.add_argument('--checkpoint-hist', type=int, default=20, metavar='N',
                    help='number of checkpoints to keep (default: 10)')

## using spikingjelly LIF
parser.add_argument('--use_spikingjelly_lif', action='store_true', default=False,
                    help="use spikingjelly LIF")


## recurrent-coding
## changed on 2025-04-08
parser.add_argument('--recurrent_coding', action='store_true', default=False)
parser.add_argument('--recurrent_lif', type=str, default=None, help="lif, plif, or None")
## 3d position embedding
parser.add_argument('--pe_type', default=None, 
                    help="position embedding methods")

args = parser.parse_args()

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
def train(model, device, train_loader, criterion, optimizer, epoch, args, use_spikingjelly_lif=False):
    running_loss = 0
    start_time = time.time()
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    r = np.random.rand(1)

    # for  i,(images, labels) in enumerate(train_loader):
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
        if use_spikingjelly_lif:
            functional.reset_net(model)

        optimizer.zero_grad()

        labels = labels.to(device)
        images = images.to(device)
        if args.beta > 0 and r < args.cutmix_prob:
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(images.size()[0]).cuda()
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                # compute output
            outputs = model(images)
            mean_out = outputs.mean(1)
            loss = criterion(mean_out, target_a) * lam + criterion(mean_out,target_b) * (1. - lam)
        else:
            # compute output
            outputs = model(images)
            mean_out = outputs.mean(1)
            loss = criterion(mean_out, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss / total, 100 * correct / total

@torch.no_grad()
def test(model, test_loader, device, use_spikingjelly_lif=False):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_spikingjelly_lif:
            functional.reset_net(model)
        inputs = inputs.to(device)
        outputs = model(inputs)
        mean_out = outputs.mean(1)
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100 * correct / total
    return final_acc

if __name__ == '__main__':



    seed_all(args.seed)
    train_dataset, val_dataset = data_loaders.build_cifar(use_cifar10=args.use_cifar10, data_path=args.data_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)


    parallel_model =  msresnet18(num_classes=100, using_GAC=args.GAC, time_step=args.time,
                                 use_spikingjelly_lif=args.use_spikingjelly_lif, 
                                 recurrent_coding=args.recurrent_coding, recurrent_lif=args.recurrent_lif,   # changed on 2025-05-23
                                 pe_type=args.pe_type)                                                       # changed on 2025-05-23
    # parallel_model.T = args.time                                        # changed on 2025-05-23

    parallel_model = torch.nn.DataParallel(parallel_model)

    parallel_model.to(device)
    # load pretain_model
#     state_dict = torch.load('.pth')
#     parallel_model.module.load_state_dict(state_dict, strict=False)

    # changed on 2025-05-23
    best_metric = None
    eval_metric = args.eval_metric
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    if args.experiment:
            exp_name = args.experiment
    else:
        assert False, "Must provide experiment name!"
    output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
    decreasing = True if eval_metric == 'loss' else False

    logger = get_logger(os.path.join(output_dir, 'train.log'))
    logger.info('start training!')

    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    optimizer = torch.optim.SGD(parallel_model.parameters(),lr=args.lr,momentum=0.9,weight_decay=5e-5)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)



    # create output folder to save tain log and checkpoint, changed on 2025-05-23
    saver = CheckpointSaver(
        model=parallel_model, optimizer=optimizer, args=args, model_ema=None, amp_scaler=None,
        checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    best_acc = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        loss, acc = train(parallel_model, device, train_loader, criterion, optimizer, epoch, args, use_spikingjelly_lif=args.use_spikingjelly_lif)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , args.epochs, loss, acc ))
        scheduler.step()
        facc = test(parallel_model, test_loader, device, use_spikingjelly_lif=args.use_spikingjelly_lif)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch , args.epochs, facc ))

        if output_dir is not None:
            update_summary(
                    epoch, OrderedDict([('loss', loss), ('acc', acc)]), OrderedDict([('top1', facc)]), os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=False)

        # if best_acc < facc:
        #     best_acc = facc
        #     best_epoch = epoch + 1
        #     torch.save(parallel_model.module.state_dict(), '.pth')
        if saver is not None:
            # save proper checkpoint with eval metric
            save_metric = facc
            best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
            logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

        # logger.info('Best Test acc={:.3f}'.format(best_acc ))
        # logger.info('\n')

