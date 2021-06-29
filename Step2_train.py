import numpy as np
import torch
import yaml
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch import device, save
from model import BeatTrackingNet
from MGTV_dataload import TrainDataset
from torch.utils.data import DataLoader
import pdb
import utils
import os
import logging
NET_MAME = 'TCN'
########################################## writer log ####################################################
if not os.path.exists('./log/'):
    os.makedirs('./log/')

log_file_name = './log/' + NET_MAME + '_log.txt'
logging.basicConfig(level=logging.INFO, format='%(asctime)s  -  %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    filename=log_file_name, filemode='a')


with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

os.makedirs(config['model_folder'], exist_ok=True)
# cuda
if torch.cuda.is_available():
    GPU = True
else:
    GPU = False

is_load = False

# Training parameters
is_train = True
cross_validation = False

num_epoch = config['num_epoch']
batch_size = config['batch_size']
optimizer = config['optimizer']
learning_rate = config['learning_rate']
k_fold = config['k_fold']

# load dataset
dataset = TrainDataset()

train_dataset = TrainDataset(mode='train')

valid_dataset = TrainDataset(mode='val')
valid_loader = DataLoader(valid_dataset, batch_size = batch_size)

model = BeatTrackingNet()
parameters = model.parameters()

if GPU:
    model = model.cuda()

if optimizer == 'Adam':
    optimizer = Adam(parameters, lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, threshold=0.0001)

criterion_bce = nn.BCELoss().cuda()

params = list(model.parameters()) + list(criterion_bce.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print(f'Total parameters: {total_params}')
# train and valid
for i in range(1, num_epoch + 1):
    # adjust lr
    if i % 30 == 0 and i != 0:
        learning_rate = learning_rate / 5.0
        utils.AdjustLearningRate(optimizer, lr=learning_rate)

    # training
    print(f"Epoch {i}: Training Start.")
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    batch_loss = list()
    batch_step = 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    for input, label in train_loader:
        # batch_step += 1
        optimizer.zero_grad()
        if GPU:
            input, label = input.float().cuda(), label.cuda()
        output = model(input)
        loss_bce = criterion_bce(output, label)
        loss = loss_bce
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_bce += loss_bce.item()
        batch_step += 1
        # print(batch_step)
        if batch_step % 10 == 0:
            log_info = f'Epoch {i}, Step {batch_step}; ' + f'Average Train Loss {running_loss / (batch_step * batch_size)}.'
            logging.info(log_info)
            print(f'Epoch {i}, Step {batch_step};  Total Step {len(train_loader)}; lr {learning_rate}; '
                  f'Average Train Loss {running_loss / (batch_step * batch_size):.6f}; '
                  f'Average bce Loss {running_bce / (batch_step * batch_size):.6f}; ')

    # validation
    model.eval()
    print(f"Epoch {i}: Validation Start...")
    train_loader = DataLoader(train_dataset, batch_size=len(valid_dataset))
    with torch.no_grad():
        for input, label in valid_loader:
            if GPU:
                input, label = input.float().cuda(), label.cuda()
            output = model(input)
            loss_bce = criterion_bce(output, label)
            log_info = f'Average Valid bce {loss_bce.item() / len(valid_dataset)}.'
            logging.info(log_info)
            print(f'Average Valid bce {loss_bce.item() / len(valid_dataset):.6f}.')
            break

    # save model
    if i % 2 == 0:
        torch.save(model.cpu().state_dict(), f"{config['model_folder']}_Epoch{i}.pt")
        if GPU:
            model.cuda()
