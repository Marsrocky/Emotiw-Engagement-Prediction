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
from OpenfaceDataset import OpenfaceDataset, OpenfaceTestset
from mil_lstm import mil_regression
from sklearn.metrics import mean_squared_error as mse


# super parameters
batch_size = 4
n_workers = 2
use_cuda = torch.cuda.is_available()
feature_num = 9
model_path = 'parameter/mil_lstm_364.pkl'

# load data
print('Load data...')
test_dataset = OpenfaceTestset(case='final')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=n_workers)

def load_net(model, path):
    model.load_state_dict(torch.load(path))
    print('[INFO] Checkpoint {} loaded'.format(path))

def test():
    ''' Test the Model '''
    # model
    model = mil_regression(feature_num=feature_num).cuda()
    # print (model)
    load_net(model, model_path)

    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    total_loss = 0
    result = []
    ground_truth = []
    file_name = []
    for inputs,name in test_loader:
        data = torch.zeros((1, 10, feature_num)).cuda()
        for i in range(10):
            for j in range(1):
                data[j,i,:] = inputs[i][j]

        outputs = model(data)
        result.append(outputs.item())
        # ground_truth.append(y.item())
        file_name.append(name)

    # transformation to [0, 0.33, 0.66, 1]
    # for i in range(len(result)):
    #     # print (result[i], ground_truth[i])
    #     if result[i] < 0.4:
    #         result[i] = 0.
    # print (mse(result, ground_truth))

    text_file = open('9_364.txt', 'w')
    for i in range(len(result)):
        text_file.write(str(result[i])  + file_name[i][0].strip('Open_face_output').strip('.txt')
         + ' ' + str(result[i]) + '\n')
    text_file.close()


def main():
    test()

if __name__ == '__main__':
    main()
