#!/usr/bin/env python
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import models
from configs import get_config
from datasets import data_aal90 as data_loader
from datasets.data_aal90 import get_data2

cfg = get_config('configs/config.yaml')
n_category = cfg["n_category"]
num_nodes = cfg["num_nodes"]


def train(model, idx_train, train_loader, val_loader, idx_val, criterion, optimizer, scheduler, device,
          num_epochs, print_freq):
    print("------------------------------------------train------------------------------------------------")
    since = time.time()
    state_dict_updates = 0
    model = model.cuda()
    model_wts_best_val_acc = copy.deepcopy(model.state_dict())
    model_wts_lowest_val_loss = copy.deepcopy(model.state_dict())
    # lr = scheduler.get_lr()
    best_acc = 0.0
    loss_min = 10000000.0
    acc_epo = 0
    loss_epo = 0
    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print(f'======Epoch {epoch}======')
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0
                idx = idx_train if phase == 'train' else idx_val
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    iter_loss = 0
                    iter_corr = 0
                    if phase == 'train':
                        for Iters, (indx, x, adj, graph, edge, y, nodeinds) in enumerate(train_loader):
                            if x.is_cuda:
                                pass
                            else:
                                x = x.cuda()
                            if y.is_cuda:
                                pass
                            else:
                                y = y.cuda()
                            if adj.is_cuda:
                                pass
                            else:
                                adj = adj.cuda()
                            outputs = model(ids=nodeinds, feats=x, edge=edge, g=graph, adja=adj, ite=Iters)
                            outputs1 = torch.sigmoid(outputs)
                            outputs1 = torch.unsqueeze(outputs1, 0)
                            _, preds = torch.max(outputs1, 1)
                            loss = criterion(outputs1, y[0])
                            corr = torch.sum(preds == y[0])
                            # 正则化
                            for n, _module in model.named_modules():
                                para_t = 0
                                # if isinstance(_module, nn.Conv1d) and (not 'downsample' in n):
                                #     p = _module.weight
                                #     p = p.reshape(p.shape[0], p.shape[1], p.shape[2])
                                #     # group lasso regularization
                                #     para_t += 0.5 * torch.sum(torch.sqrt(torch.sum(torch.sum(p ** 2, 0), 0))).double()
                                #     # exclusive sparsity regularization
                                #     para_t += (1 - 0.5) * torch.sum((torch.sum(torch.sum(torch.abs(p), 0), 1)) ** 2).double()
                                if isinstance(_module, nn.Linear) and (not 'downsample' in n):
                                    p = _module.weight
                                    p = p.reshape(p.shape[0], p.shape[1])
                                    # regularization
                                    para_t += 0.5 * torch.sum(
                                        torch.sqrt(torch.sum(torch.sum(p ** 2, 0), 0))).double()
                                    para_t += (1 - 0.5) * torch.sum(
                                        (torch.sum(torch.sum(torch.abs(p), 0), 0)) ** 2).double()
                                loss += optimizer.defaults['weight_decay'] * para_t
                            iter_loss += loss
                            iter_corr += corr
                    else:
                        for Iters, (indx, x, adj, graph, edge, y, nodeinds) in enumerate(val_loader):
                            if x.is_cuda:
                                pass
                            else:
                                x = x.cuda()
                            if y.is_cuda:
                                pass
                            else:
                                y = y.cuda()
                            if adj.is_cuda:
                                pass
                            else:
                                adj = adj.cuda()
                            outputs = model(ids=nodeinds, feats=x, edge=edge, g=graph, adja=adj, ite=Iters)
                            outputs1 = torch.sigmoid(outputs)
                            outputs1 = torch.unsqueeze(outputs1, 0)
                            _, preds = torch.max(outputs1, 1)
                            loss = criterion(outputs1, y[0])
                            corr = torch.sum(preds == y[0])
                            for n, _module in model.named_modules():
                                para_t = 0
                                # if isinstance(_module, nn.Conv1d) and (not 'downsample' in n):
                                #     p = _module.weight
                                #     p = p.reshape(p.shape[0], p.shape[1], p.shape[2])
                                #     # group lasso regularization
                                #     para_t += 0.5 * torch.sum(torch.sqrt(torch.sum(torch.sum(p ** 2, 0), 0))).double()
                                #     # exclusive sparsity regularization
                                #     para_t += (1 - 0.5) * torch.sum((torch.sum(torch.sum(torch.abs(p), 0), 1)) ** 2).double()
                                if isinstance(_module, nn.Linear) and (not 'downsample' in n):
                                    p = _module.weight
                                    p = p.reshape(p.shape[0], p.shape[1])
                                    # regularization
                                    para_t += 0.5 * torch.sum(
                                        torch.sqrt(torch.sum(torch.sum(p ** 2, 0), 0))).double()
                                    # regularization
                                    para_t += (1 - 0.5) * torch.sum(
                                        (torch.sum(torch.sum(torch.abs(p), 0), 0)) ** 2).double()
                                loss += optimizer.defaults['weight_decay'] * para_t
                            iter_loss += loss
                            iter_corr += corr

                    running_loss += iter_loss
                    running_corrects += iter_corr

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        iter_loss.backward()
                        optimizer.step()

                # statistics
                epoch_loss = running_loss.double() / len(idx)
                epoch_acc = running_corrects.double() / len(idx)

                if epoch % print_freq == 0:
                    print(f'{phase} Acc: {epoch_acc:.4f}, Loss: {epoch_loss:.4f} ')
                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        model_wts_best_val_acc = copy.deepcopy(model.state_dict())
                        acc_epo = epoch
                        state_dict_updates += 1

                    if phase == 'val' and epoch_loss < loss_min:
                        loss_min = epoch_loss
                        model_wts_lowest_val_loss = copy.deepcopy(model.state_dict())
                        loss_epo = epoch
                        state_dict_updates += 1

                    if epoch % print_freq == 0 and phase == 'val':
                        print(f'Best val Acc: {best_acc:4f}, Min val loss: {loss_min:4f}')
                        print('-' * 60)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'\nState dict updates {state_dict_updates}')
    print(f'Best val Acc: {best_acc:4f}')
    return (model_wts_best_val_acc, acc_epo), (model_wts_lowest_val_loss, loss_epo)


def test(model, best_model_wts, idx_test, test_loader, device, test_time=5):
    best_model_wts, epo = best_model_wts
    model = model.cuda()
    model.load_state_dict(best_model_wts)
    model.eval()
    running_corrects = 0

    for t in range(test_time):
        for Iters, (indx, x, adj, graph, edge, y, nodeinds) in enumerate(test_loader):
            with torch.no_grad():
                if x.is_cuda:
                    pass
                else:
                    x = x.cuda()
                if y.is_cuda:
                    pass
                else:
                    y = y.cuda()
                if adj.is_cuda:
                    pass
                else:
                    adj = adj.cuda()
                outputs = model(nodeinds, x, edge, graph, adj, Iters)
                outputs1 = torch.sigmoid(outputs)
                outputs1 = torch.unsqueeze(outputs1, 0)
                _, preds = torch.max(outputs1, 1)
                running_corrects += torch.sum(preds == y[0]).item()

        test_acc = 1.0 * running_corrects / (len(idx_test) * (t + 1))
        print(f'Test acc: {test_acc} @Epoch-{epo}')
    return test_acc, epo


def train_test_model(cfg):
    device = torch.device('cuda:0')
    source = source_select()

    print(f'Using {cfg["activate_dataset"]} dataset')
    _, idx_train, idx_val, idx_test, _, _, _, _ = source

    print('get data okay')
    train_loader = Data.DataLoader(data_loader.brains_loader(idx_train), batch_size=1, shuffle=True,
                                   num_workers=0, pin_memory=True, drop_last=False)
    print('len of train_loader:', len(train_loader))
    val_loader = Data.DataLoader(data_loader.brains_loader(idx_val), batch_size=1, shuffle=True,
                                 num_workers=0, pin_memory=True, drop_last=False)
    print('len of val_loader:', len(val_loader))
    test_loader = Data.DataLoader(data_loader.brains_loader(idx_test), batch_size=1, shuffle=True,
                                  num_workers=0, pin_memory=True, drop_last=False)
    print('len of test_loader:', len(test_loader))
    print('load split data okay')

    model = models.model_factory.model_select(cfg['model']) \
        (dim_feat=cfg["num_nodes"],
         n_categories=cfg["n_category"],
         k_structured=cfg['k_structured'],
         k_nearest=cfg['k_nearest'],
         k_cluster=cfg['k_cluster'],
         wu_knn=cfg['wu_knn'],
         wu_kmeans=cfg['wu_kmeans'],
         wu_struct=cfg['wu_struct'],
         clusters=cfg['clusters'],
         adjacent_centers=cfg['adjacent_centers'],
         n_layers=cfg['n_layers'],
         layer_spec=cfg['layer_spec'][0],
         dropout_rate=cfg['drop_out'],
         has_bias=cfg['has_bias']
         )

    state_dict = model.state_dict()
    for key in state_dict:
        if 'weight' in key:
            # nn.init.xavier_uniform_(state_dict[key])
            state_dict[key] = state_dict[key].tolist()

        elif 'bias' in key:
            state_dict[key] = state_dict[key].zero_()

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], eps=1e-20)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg['milestones'],
                                               gamma=cfg['gamma'])
    criterion = torch.nn.CrossEntropyLoss()

    # learning mode
    model.cuda()
    print('model okay')

    # train
    model_wts_best_val_acc, model_wts_lowest_val_loss \
        = train(model, idx_train, train_loader, val_loader, idx_val, criterion, optimizer, schedular, device,
                cfg['max_epoch'], cfg['print_freq'])

    # test
    if idx_test is not None:
        print('**** Model of lowest val loss ****')
        test_acc_lvl, epo_lvl \
            = test(model, model_wts_lowest_val_loss, idx_test, test_loader, device, cfg['test_time'])
        print('**** Model of best val acc ****')
        test_acc_bva, epo_bva \
            = test(model, model_wts_best_val_acc, idx_test, test_loader, device, cfg['test_time'])
        return (test_acc_lvl, epo_lvl), (test_acc_bva, epo_bva)
    else:
        return None


def source_select():
    return get_data2(cfg)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg["gpu_id"])
    setup_seed(cfg["seed_num"])
    (test_acc_lvl, epo_lvl), (test_acc_bva, epo_bva) = train_test_model(cfg)
