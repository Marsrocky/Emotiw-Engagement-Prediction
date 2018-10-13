import math
import argparse
import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from torch import optim
from torch.autograd import Variable 
from torchvision import transforms
from OpenfaceDataset import OpenfaceDataset
from mil_lstm import mil_regression

# super parameters
batch_size = 4
n_workers = 2
num_epochs = 40
lr = 0.001
weight_decay = 5e-4
use_cuda = torch.cuda.is_available()
level = [0., 0.33, 0.66, 1.]
feature_num = 9

# load data
print('Load data...')
train_dataset = OpenfaceDataset(case='train')
test_dataset = OpenfaceDataset(case='test')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=n_workers)

# model
model = mil_regression(feature_num=feature_num).cuda()
print (model)
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30], gamma=0.1)
loss_history = {"train": [], "test": []}

def save_net(model, path):
    torch.save(model.state_dict(), path)
    print('[INFO] Checkpoint saved to {}'.format(path))


def load_net(model, path):
    model.load_state_dict(torch.load(path))
    print('[INFO] Checkpoint {} loaded'.format(path))

def train(epoch):
    scheduler.step()
    model.train()
    total = 0
    correct = 0
    total_loss = 0
    for i, (inputs, targets) in enumerate(train_loader):
        targets = Variable(targets.float()).cuda()

        data = torch.zeros((batch_size, 10, feature_num)).cuda()
        for i in range(10):
            for j in range(batch_size):
                data[j,i,:] = inputs[i][j]
        
        optimizer.zero_grad()
        outputs = model(data)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total += targets.size(0)
        total_loss += loss

        loss_history['train'].append(loss)
        if i % 30 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] MSELoss: %.4f' % (epoch+1, num_epochs, i+1, len(train_loader), loss))
    print('Epoch [%d], Average Training MSEloss: %.4f' % (epoch+1, total_loss/total))

def test(epoch):
    ''' Test the Model '''
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    total_loss = 0
    for inputs,targets in test_loader:
        targets = Variable(targets.float()).cuda()

        data = torch.zeros((1, 10, feature_num)).cuda()
        for i in range(10):
            for j in range(1):
                data[j,i,:] = inputs[i][j]

        outputs = model(data)
        loss = criterion(outputs, targets)
        total += targets.size(0)
        # _, predicted = torch.max(outputs.data,1)
        # _, labels = torch.max(targets.data, 1)
        # correct += predicted.eq(labels).cpu().sum()
        total_loss += loss
    
    if len(loss_history['test']) == 0:
        save_net(model, 'parameter/mil_lstm.pkl')
    elif total_loss/total < min(loss_history['test']):
        save_net(model, 'parameter/mil_lstm.pkl')

    loss_history['test'].append(total_loss/total)
    print('Epoch [%d], Average Test MSEloss: %.4f' % (epoch+1, total_loss/total))



def main():
    for epoch in xrange(num_epochs):
        train(epoch)
        test(epoch)
        print('Best test MLE loss: %.4f' % min(loss_history['test']))

if __name__ == '__main__':
    main()
