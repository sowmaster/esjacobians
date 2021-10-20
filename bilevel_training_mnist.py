import os
from networks import LeNet5Feats, ResNetFeats18, classifier
#import resnet
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import higher
import hypergrad as hg
from utils import save_checkpoint
import time
import matplotlib.pyplot as plt
import pickle

#os.environ["CUDA_VISIBLE_DEVICES"] = '2'


parser = argparse.ArgumentParser(description='Bilevel Training')

# Basic model parameters.
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'cifar10'])
parser.add_argument('--data', type=str, default='./data')
parser.add_argument('--output_dir', type=str, default='hyperlogs')
args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

if args.dataset == 'MNIST':
    data_test = MNIST(args.data,
                       download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    data_train = MNIST(args.data,
                      train=False,
                      download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ]))

    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=4)
    data_test_loader = DataLoader(data_test, batch_size=256, shuffle=True, num_workers=4)

    hypernet = LeNet5Feats().cuda()
    cnet = classifier(n_features=84, n_classes=10).cuda()
    lr = 0.001

if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_test = CIFAR10(args.data,
                         transform=transform_train,
                         download=True)
    data_train = CIFAR10(args.data,
                        train=False,
                        transform=transform_test,
                        download=True)

    data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=4)
    data_test_loader = DataLoader(data_test, batch_size=128, shuffle=True, num_workers=4)

    hypernet = ResNetFeats18().cuda()
    cnet = classifier(n_features=512, n_classes=10).cuda()
    lr = 0.05

print('learning rate is = ', lr)
numtest = len(data_test_loader)
print('num of outer batches = ', numtest)
numtrain = len(data_train_loader)
print('num of inner batches = ', numtrain)

data_train_iter = iter(data_train_loader)
data_test_iter = iter(data_test_loader)

fhnet = higher.monkeypatch(hypernet, copy_initial_weights=True).cuda()
hparams = list(hypernet.parameters())
hparams = [hparam.requires_grad_(True) for hparam in hparams]
fcnet = higher.monkeypatch(cnet, copy_initial_weights=True).cuda()
params = list(cnet.parameters())
params = [param.requires_grad_(True) for param in params]

numhparams = sum([torch.numel(hparam) for hparam in hparams])
numparams = sum([torch.numel(param) for param in params])

print('size of outer variable: ', numhparams)
print('size of inner variable: ', numparams)


init_params = []
for param in params:
    init_params.append(torch.zeros_like(param))

criterion = torch.nn.CrossEntropyLoss().cuda()
outer_opt = torch.optim.Adam(hparams, lr=lr)

def outer_loss(params, hparams, more=False):

    global data_test_iter

    try:
        images, labels = next(data_test_iter)
    except StopIteration:
        data_test_iter = iter(data_test_loader)
        images, labels = next(data_test_iter)

    images, labels = images.cuda(), labels.cuda()

    feats = fhnet(images, params=hparams)
    outputs = fcnet(feats, params=params)
    loss = criterion(outputs, labels)

    preds = outputs.data.max(1)[1]
    correct = preds.eq(labels.data.view_as(preds)).sum()
    acc = float(correct) / labels.size(0)

    if more:
        return loss, acc
    else:
        return loss

def update_tensor_grads(params, grads):
    for l, g in zip(params, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g

def inner_solver(hparams, hparams_mu, steps, params0=None):

    global data_train_iter

    if params0 is not None:
        for param, param0 in zip(params, params0):
            param.data = param0.data

    params_mu = [p.detach().clone() for p in params]  #
    params_mu = [p.requires_grad_(True) for p in params_mu]

    optim = torch.optim.Adam(params=params+params_mu, lr=lr)


    for i in range(steps):
        try:
            images, labels = next(data_train_iter)
        except StopIteration:
            data_train_iter = iter(data_train_loader)
            images, labels = next(data_train_iter)


        images, labels = images.cuda(), labels.cuda()

        optim.zero_grad()
        feats = fhnet(images, params=hparams)
        outputs = fcnet(feats, params=params)
        loss = criterion(outputs, labels)
        grads = torch.autograd.grad(loss, params)
        update_tensor_grads(params, grads)
        optim.step()

        optim.zero_grad()
        feats = fhnet(images, params=hparams_mu)
        outputs = fcnet(feats, params=params_mu)
        loss = criterion(outputs, labels)
        grads = torch.autograd.grad(loss, params_mu)
        update_tensor_grads(params_mu, grads)
        optim.step()

    return [[p.detach() for p in params], [p.detach() for p in params_mu]]

run = 1
T = 10 # 10 is good
mu = 0.01 # 0.1 is good
steps = 1000
warm_start = True # if set True, T can be low and it's fast!
total_time = 0
running_time, outer_accs, outer_losses = [], [], []
# avg_acc = 0.0
# count = 0

print('Bilevel training with ESJ on ' + args.dataset)
print('Number of inner iterations T=', T)
print('Smoothness parameter mu=', mu)

for step in range(1, steps+1):
    t0 = time.time()
    us = [torch.randn(hparam.size()).cuda() for hparam in hparams]
    us = [u / torch.norm(u, 2) for u in us]
    hparams_mu = [mu * u + hparam for u, hparam in zip(us, hparams)]

    params_list = inner_solver(hparams, hparams_mu, T, params0=init_params)

    outer_opt.zero_grad()

    _, oloss, oacc= hg.sczoj(params_list, hparams, us, outer_loss, mu, set_grad=True, more=True)
    # avg_acc += oacc
    outer_opt.step()

    t1 = time.time() - t0
    total_time += t1
    # count += 1
    running_time.append(total_time)
    outer_accs.append(oacc)
    outer_losses.append(oloss.item())


    for init_p, p in zip(init_params, params_list[0]):
        if warm_start:
            init_p.data = p
        else:
            init_p.data = torch.zeros_like(init_p)

    if step % 10 == 0:
        print('Step: %d/%d, lr: %f, Outer Batch Loss: %f, Accuracy on outer batch: %f' % (step, steps, lr, oloss.item(), oacc))

    if step == 80 and args.dataset == 'MNIST':
        lr = 0.0005
        for param_group in outer_opt.param_groups:
            param_group['lr'] = lr

    if step == 180 and args.dataset == 'MNIST':
        lr = 0.0001
        for param_group in outer_opt.param_groups:
            param_group['lr'] = lr

    # if step == 25000 and args.dataset == 'cifar10':
    #     lr = 0.005
    #     for param_group in outer_opt.param_groups:
    #         param_group['lr'] = lr
    #
    # if step == 40000 and args.dataset == 'cifar10':
    #     lr = 0.0005
    #     for param_group in outer_opt.param_groups:
    #         param_group['lr'] = lr

print('Ended in {:.2e} seconds\n'.format(total_time))

filename = 'ESJ_' + args.dataset + '_T' + str(T) + '_run' + str(run) + '.pt'
save_path = os.path.join(args.output_dir, filename)

state_dict = {'runtime': running_time,
              'accuracy': outer_accs,
              'loss': outer_losses}

save_checkpoint(state_dict, save_path)

# with open(os.path.join("hyperlogs", filename), "ab") as f:
#     pickle.dump([running_time, outer_accs, outer_losses], f)