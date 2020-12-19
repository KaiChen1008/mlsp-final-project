from __future__ import print_function

import os
import argparse
import csv
import math
import scipy.stats as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from utils import *
from models.resnet_model import resnet18
from models.smooth_resnet_model import smooth_resnet18

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.attacks import LinfPGDAttack, PGDAttack


parser = argparse.ArgumentParser( description='PyTorch adversarial training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess',         default='default',   type=str,   help='session id')
parser.add_argument('--seed',         default=86108,       type=int,   help='rng seed')
parser.add_argument('--lr',           default=0.1,         type=float, help='initial learning rate')
parser.add_argument('--batch-size', '-b', default=256,     type=int,   help='mini-batch size (default: 256)')
parser.add_argument('--epochs',       default=10,          type=int,   help='number of total epochs to run')
parser.add_argument('--eps',          default=0.5,         type=float, help='perturbation')
parser.add_argument('--iter',         default=7,          type=int,   help='adv iteration')
parser.add_argument('--smooth',       action='store_true', help='use smooth model')
parser.add_argument('--adv-train',    action='store_true', help='adv train')
args = parser.parse_args()

torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
batch_size = args.batch_size
base_learning_rate = args.lr
is_training = True


# Data (Default: MNIST)
print('==> Preparing MNIST data.. (Default)')

# scale to [0, 1] without standard normalize
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

train_set = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=16)

test_set = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=16)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if args.smooth:
        checkpoint = torch.load('./checkpoint/ckpt.t7.' + args.sess + '_' + str(args.seed)+ '_smooth')
    else:
        checkpoint = torch.load('./checkpoint/ckpt.t7.' + args.sess + '_' + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
else:
    print('==> Building model.. (Default :resnet18)')
    print('==> Is smooth model : ', args.smooth)
    start_epoch = 0
    if args.smooth:
        net = smooth_resnet18(10)
    else:
        net = resnet18(10)
    

result_folder = './results/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

logname = result_folder + net.__class__.__name__ + \
    '_' + args.sess + '_' + str(args.seed) + '.csv'

if use_cuda:
    net.cuda()
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.99))

# Training
if args.adv_train:
    print('==> Start adversarial training..')
else:
    print('==> Start normal training..')
def train(epoch):
    global is_training 
    is_training = True
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda().requires_grad_(), targets.cuda()

        # generate adv img
        if args.adv_train:
            adversary = PGDAttack(
                net, 
                loss_fn=nn.CrossEntropyLoss(reduction="sum"), 
                eps=args.eps ,
                nb_iter=args.iter, 
                eps_iter=args.eps/args.iter, 
                rand_init=True, 
                clip_min=0.0,
                clip_max=1.0, 
                targeted=False)
            with ctx_noparamgrad_and_eval(net):
                inputs = adversary.perturb(inputs, targets)
            
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return (train_loss / batch_idx, 100. * correct / total)


def test(epoch,is_adv=False):
    global is_training, best_acc
    is_training = False
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    # with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.requires_grad_().cuda(), targets.cuda()
        if is_adv:
            adversary = PGDAttack(
                net, 
                loss_fn=nn.CrossEntropyLoss(reduction="sum"), 
                eps=args.eps ,
                nb_iter=args.iter, 
                eps_iter=args.eps/args.iter, 
                rand_init=True, 
                clip_min=0.0,
                clip_max=1.0, 
                targeted=False)
            
            with ctx_noparamgrad_and_eval(net):
                inputs = adversary.perturb(inputs, targets)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc and is_adv == False:
        best_acc = acc
        checkpoint(acc, epoch)
    return (test_loss / batch_idx, 100. * correct / total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if args.smooth:
        torch.save(state, './checkpoint/ckpt.t7.' + args.sess + '_' + str(args.seed)+ '_smooth')
    else:
        torch.save(state, './checkpoint/ckpt.t7.' + args.sess + '_' + str(args.seed))


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(
            ['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

for epoch in range(start_epoch, args.epochs):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch, is_adv=True)
    test_loss, test_acc = test(epoch, is_adv=False)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
