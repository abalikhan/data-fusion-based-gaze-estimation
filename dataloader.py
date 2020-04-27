import os
import numpy as np
import deepdish as dd
import torch
import h5py
import torch.utils.data as udata
from torchvision import transforms
from sklearn.utils import shuffle
import cv2
import scipy.io as sio

class MPIIGazeDataset(udata.Dataset):
    def __init__(self, dataset_dir, subject_id, data_type='train'):
        path = os.path.join(dataset_dir, '{}.h5'.format(subject_id))

        data = dd.io.load(path)
        self.L_images = data['leye']
        self.R_images = data['reye']
        self.headpose = data['headpose']
        self.gazes = data['gaze']
        # data = h5py.File(path)
        # self.L_images = data[data_type+'/imagesL']
        # self.R_images = data[data_type+'/imagesR']
        # self.headpose = data[data_type+'/headposes']
        # self.gazes = data[data_type+'/gazes']
        self.outpath = r'D:\PycharmProjects\shallow_network\saved 3 (good model)\images checking\\'

        # self.L_images, self.R_images, self.headpose, self.gazes = shuffle(self.L_images, self.R_images, self.headpose, self.gazes)

        self.length = len(self.L_images)


        self.eye_transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomCrop(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                  std=[0.229, 0.224, 0.225])
        ])


    def __getitem__(self, index):
        # left eye transform
        imLeye = self.L_images[index]
        imLeye = imLeye.reshape(80, 80, 3)
        imLeye = cv2.resize(imLeye, dsize=(60, 60), interpolation=cv2.INTER_AREA)
        cv2.imwrite(self.outpath+'Leye_%05d.jpg'%index, imLeye)
        imLeye = np.transpose(imLeye, (2, 1, 0))

        imLeye = torch.as_tensor(imLeye, dtype=torch.float)
        # imLeye = imLeye.transpose(2, 0)

        imReye = self.R_images[index]
        imReye = imReye.reshape(80, 80, 3)
        imReye = cv2.resize(imReye, dsize=(60, 60), interpolation=cv2.INTER_AREA)/255
        cv2.imwrite(self.outpath+'Reye_%05d.jpg'%index, imReye)
        imReye = np.transpose(imReye, (2, 1, 0))
        imReye = torch.as_tensor(imReye, dtype=torch.float)
        # imReye = imReye.transpose(2, 0)


        # applying transformation
        # imLeye = self.eye_transform(imLeye)
        # imReye = self.eye_transform(imReye)

        #headpose
        headpose = self.headpose[index]
        headpose = torch.as_tensor(headpose, dtype=torch.float)

        # X = [imLeye, imFace, headpose, imlandmark]
        gaze = self.gazes[index]
        gaze = torch.as_tensor(gaze, dtype=torch.float)
        return imLeye, imReye, headpose, gaze

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__


def get_loader(data_path, subjects, d_type='train'):
    assert os.path.exists(data_path)

    subject_ids = ['%2s'%subject for subject in zip(subjects)]
    # subject_id = r'RT_GENE_test_s009'
    dataset = udata.ConcatDataset([
        MPIIGazeDataset(data_path, subject_id, data_type=d_type) for subject_id in subject_ids
    ])

    return dataset


def vector2angles(gaze_vector):

    """

    Transforms a gaze vector into the angles yaw and elevation/pitch.

    :param gaze_vector: 3D unit gaze vector

    :return: 2D gaze angles

    """

    gaze_angles = np.empty((1, 2), dtype=np.float32)

    gaze_angles[0, 0] = np.arctan(-gaze_vector[0]/-gaze_vector[2])  # phi= arctan2(x/z)

    gaze_angles[0, 1] = np.arcsin(-gaze_vector[1])  # theta= arcsin(y)

    return gaze_angles

import matplotlib.pyplot as plt
if __name__ == '__main__':
    import h5py
    data_path = r'D:\RT-Gene_Dataset\non_augment\\'
    data1 = sio.loadmat(data_path + 'RT_GENE_train_s000.mat')
    data2 = sio.loadmat(data_path + 'RT_GENE_test_s000.mat')
    print('sizes are %d'%(np.shape(data1['train/imagesL'])) +'and test %d' %(np.shape(data2['imagesR'])))
    # test = ['s001', 's002', 's008', 's010', 's000']
    # dataTest = get_loader(data_path, test)
    # for i, (l, r, h, g) in enumerate(dataTest):
    #     print(i)