import torch
import os
import csv
import pickle
import numpy as np
from torch.utils import data

# OpenFace_features/Train/subject_26_Vid_1.txt
class OpenfaceDataset(torch.utils.data.Dataset):
    ''' Load dataset as torch.tensor '''
    def __init__(self, case='test', root='OpenFace_features2/Train', l_dir='OpenFace_features/Labels.csv'):
        self.case=case
        self.file_list = []
        self.label_list = []
        self.level = [0., 0.33, 0.66, 1.]

        # # Only run for the first time
        # for sample in os.listdir(root):
        #     target_file = os.path.join(root, sample)
        #     self.file_list.append(target_file)
        #     with open(l_dir, 'rb') as myfile:
        #         lines = csv.reader(myfile)
        #         for line in lines:
        #             if line[0] == sample.strip('.txt'):
        #                 self.label_list.append(int(line[1]))
        #                 break
        # assert len(self.file_list) == len(self.label_list)
        # self.all_features = self.get_feature()

        # Load data to speed up
        with open(case+'_features.txt', 'rb') as fp:
            self.all_features = pickle.load(fp)
        with open(case+'_label.txt', 'rb') as fp:
            self.label_list = pickle.load(fp)
        print (case+' dataset loaded successfully!')

    def get_feature(self):
        features = []
        for idx in range(len(self.label_list)):
            # segment video to 10 segments, return features
            file_dir, label = self.file_list[idx], self.label_list[idx]
            v_data = np.genfromtxt(file_dir, delimiter=',',dtype='str')
            v_data = np.delete(v_data, 0, 0)    # delete table caption
            v_data = v_data.astype(np.float)[:, 4:13]   # gaze / pose

            # remove nan
            v_data = v_data[~np.isnan(v_data).any(axis=1)]
            print v_data.shape

            # average variance of gaze in video + std in segment
            m = np.mean(v_data, axis=0)
            interval = v_data.shape[0]/10
            feature = []
            for i in range(10):
                seg = v_data[i*interval:(i+1)*interval,:]
                gaze_f = np.sum(np.abs(seg[:,:6] - m[:6]), axis=0)
                head_f = np.std(seg, axis=0)[-3:]
                feature.append(torch.FloatTensor(np.append(gaze_f, head_f)))
            features.append(feature)
            print file_dir
        with open(self.case+'_features.txt', 'wb') as fp:
            pickle.dump(features, fp)
        with open(self.case+'_label.txt', 'wb') as fp:
            pickle.dump(self.label_list, fp)
        return features

    def __getitem__(self, idx):
        x, y = self.all_features[idx], self.level[self.label_list[idx]]
        return x, y

    def __len__(self):
        return len(self.label_list)

class OpenfaceTestset(torch.utils.data.Dataset):
    ''' Load dataset as torch.tensor '''
    def __init__(self, case='final', root='Open_face_output'):
        self.case=case
        self.file_list = []
        self.label_list = []
        self.level = [0., 0.33, 0.66, 1.]

        # # Only run for the first time
        # for sample in os.listdir(root):
        #     target_file = os.path.join(root, sample)
        #     self.file_list.append(target_file)
        # self.all_features = self.get_feature()

        # Load data to speed up
        with open(case+'_features.txt', 'rb') as fp:
            self.all_features = pickle.load(fp)
        if case == 'final':
            with open(self.case+'_file_list.txt', 'rb') as fp:
                self.file_list = pickle.load(fp)
        if case != 'final':
            with open(case+'_label.txt', 'rb') as fp:
                self.label_list = pickle.load(fp)
        print (case+' dataset loaded successfully!')
        print (len(self.all_features))
        print (len(self.file_list))

    def get_feature(self):
        features = []
        for idx in range(len(self.file_list)):
            # segment video to 10 segments, return features
            file_dir = self.file_list[idx]
            print file_dir
            v_data = np.genfromtxt(file_dir, delimiter=',',dtype='str')
            v_data = np.delete(v_data, 0, 0)    # delete table caption
            v_data = v_data.astype(np.float)[:, 4:16]   # gaze / pose

            # remove nan
            v_data = v_data[~np.isnan(v_data).any(axis=1)]
            print v_data.shape

            # average variance of gaze in video + std in segment
            m = np.mean(v_data, axis=0)
            interval = v_data.shape[0]/10
            feature = []
            for i in range(10):
                seg = v_data[i*interval:(i+1)*interval,:]
                gaze_f = np.sum(np.abs(seg[:,:6] - m[:6]), axis=0)
                head_f = np.std(seg, axis=0)[-6:]
                feature.append(torch.FloatTensor(np.append(gaze_f, head_f)))
            features.append(feature)
        with open(self.case+'_features_12.txt', 'wb') as fp:
            pickle.dump(features, fp)
        with open(self.case+'_file_list_12.txt', 'wb') as fp:
            pickle.dump(self.file_list, fp)
        return features

    def __getitem__(self, idx):
        if self.case == 'final':
            x = self.all_features[idx]
            return x, self.file_list[idx]
        else:
            x, y = self.all_features[idx], self.level[self.label_list[idx]]
            return x, y, self.file_list[idx]

    def __len__(self):
        return len(self.all_features)

if __name__ == '__main__':
    # test
    train_dataset = OpenfaceDataset(case='train')
    # test_dataset = OpenfaceTestset()

    # dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=2, shuffle=True)
    # for idx, (data, target) in enumerate(dataloader):
    #     tdata = torch.zeros((12, 10, 9))
    #     for i in range(10):
    #         for j in range(12):
    #             tdata[j][i] = data[i][j]
    #     print (tdata.shape)
    #     print target
    #     break