import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from OpenfaceDataset import OpenfaceDataset

class lstm_regression(nn.Module):
    def __init__(self, feature_num=9, hidden_dim=64):
        ''' define the LSTM regression network '''
        super(lstm_regression, self).__init__()
        # self.lstm = nn.LSTM(feature_num, hidden_dim, 2, batch_first=True, dropout=0.5)
        self.lstm = nn.LSTM(feature_num, hidden_dim, 1, batch_first=True)

        self.dense = torch.nn.Sequential(
            nn.Linear(hidden_dim, 1028),
            nn.Linear(1028, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 1)
        )

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        output,_ = self.lstm(inputs)
        output = self.dense(output[:,-1,:])
        return output

class mil_regression(nn.Module):
    def __init__(self, feature_num=9, hidden_dim=64):
        ''' use LSTM regression for MIL '''
        super(mil_regression, self).__init__()
        self.seg_num = 10
        self.net = lstm_regression(feature_num, hidden_dim)

    def forward(self, inputs):
        self.b, self.seg_num, _ = inputs.shape
        outputs = torch.zeros((self.b, self.seg_num)).cuda() # 12 * 10
        for i in range(10):
            outputs[:,i] = self.net(inputs[:,i,:]).squeeze()
        # for idx, seg in enumerate(inputs):
        #     seg = Variable(seg).cuda()
        #     outputs[idx] = self.net(seg)
        output = torch.mean(outputs, 1).cuda()
        return output


if __name__ == '__main__':
    net = mil_regression().cuda()
    dataset = OpenfaceDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
    for idx, (data, target) in enumerate(train_loader):
        output = net(data)
        print (output)
        break
