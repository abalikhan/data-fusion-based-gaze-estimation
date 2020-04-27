import os
import scipy.io as sio
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import h5py
import cv2


mean = [129.186279296875, 104.76238250732422,93.59396362304688]
std = [1, 1, 1]
class MPIIFaceGazeDataset(torch.utils.data.Dataset):
    def __init__(self, subject_id, dataset_dir):
        path = os.path.join(dataset_dir, '{}.h5'.format(subject_id))


        # f = sio.loadmat(path)
        f = h5py.File(path, 'r')
        # self.images = f.get('/Data/data')
        # self.gz = f.get('/Data/label')
        self.images = f.get('/Data/data')[()]
        self.gz = f.get('/Data/label')[()]
        f.close()
        # with h5py.File(path, 'r') as f:
        #
        #     self.images = f.get('/Data/data')[()]
        #     self.gz = f.get('/Data/label')[()]
        #     f.close()

        # self.images = f['faces']
        # self.gz = f['labels']

        self.length = len(self.images)
        # self.data_transform = transforms.Compose([
        #
        #                                     transforms.Resize((224, 224)),
        #                                     transforms.RandomCrop(256),
        #                                     transforms.ToTensor(),
        #                                     # transforms.Normalize([[129.186279296875, 104.76238250732422,
        #                                     #                        93.59396362304688]], [1, 1, 1])
        # ])



    def __getitem__(self, index):
        face = self.images[index].transpose(1, 2, 0)
        face = cv2.resize(face, (224, 224), cv2.INTER_CUBIC)
        face = ((face - mean)/std)/255.
        face = np.transpose(face, (2, 1, 0))

        face = torch.as_tensor(face, dtype=torch.float32)


        # face = self.data_transform(face)

        gaze = torch.as_tensor(self.gz[index][0:2], dtype=torch.float32)

        return face, gaze

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__


def get_loader(dataset_dir, test_subject_id, batch_size, num_workers):
    assert os.path.exists(dataset_dir)
    assert test_subject_id in range(15)

    subject_ids = ['p{:02}'.format(i) for i in range(0, 2)]
    test_subject_id = subject_ids[test_subject_id]

    train_dataset = torch.utils.data.ConcatDataset([
        MPIIFaceGazeDataset(subject_id, dataset_dir) for subject_id in subject_ids if subject_id != test_subject_id
    ])
    test_dataset = MPIIFaceGazeDataset(test_subject_id, dataset_dir)

    # assert len(train_dataset) == 42000
    # assert len(test_dataset) == 3000

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader

# if __name__ == '__main__':
#     path = r'D:\MPIIFaceGaze\MPIIFACE_Processed\p01.mat'

    # f = h5py.File(path, 'r')
    # with h5py.File(path, 'r') as f:

    # f = sio.loadmat(path)
    # data = f['Data/data'][()]
    # data = f['faces']
    # img = data[0].reshape(224, 224,3)
    # print(data.items())

    # print(images.shape)

    # img1 = images[1][[2, 1, 0], :, :].transpose((1, 2, 0))
    # img1 = cv2.resize(img1, (224, 224), cv2.INTER_CUBIC)

    # print()