from __future__ import print_function
import os
from data import Dataset
import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
import torchvision
from utils import Visualizer, view_model
import torch
import numpy as np
import random
import time
from config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from test import get_featurs
from sklearn.metrics import roc_auc_score

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def get_val_dataset(path):
    true_list = []
    a_path = []
    b_path = []
    with open(path) as f:
        for line in f:
            line = line.strip('\n')
            info = line.split(',')
            true_list.append(int(info[0]))
            a_path.append(info[1])
            b_path.append(info[2])
    return true_list, a_path, b_path

def val(model, true_list, a_list, b_list):
    a_features, a_cnt = get_featurs(model, a_list)
    b_features, b_cnt = get_featurs(model, b_list)
    anses = []
    #  print(a_features.shape)
    for a_f, b_f in zip(a_features, b_features):
        ans = np.dot(a_f, b_f)/(np.linalg.norm(a_f)*(np.linalg.norm(b_f)))
        anses.append(ans)
    return roc_auc_score(np.array(true_list), np.array(anses))


if __name__ == '__main__':

    opt = Config()

    device = torch.device("cuda")

    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    true_list, a_path, b_path = get_val_dataset(opt.val_list)

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model = resnet_face18(use_se=opt.use_se)

    if opt.metric == 'add_margin':
        print(f'Metric: add_margin')
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        print(f'Metric: arc_margin')
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        print(f'Metric: sphere_margin')
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)


    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        print(f'optimizer: sgd')
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        print(f'optimizer: Adam')
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)

    start = time.time()
    for i in range(opt.max_epoch):
        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                
                start = time.time()

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.save_name, i)
            save_model(metric_fc, opt.checkpoints_path, f"{opt.save_name}_metric", i)
        model.eval()
        acc = val(model, true_list, a_path, b_path)

        print('{} train epoch {} val acc {}'.format(time_str, i, acc))
        scheduler.step()
