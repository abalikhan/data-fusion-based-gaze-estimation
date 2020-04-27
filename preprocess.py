import h5py
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess(dataset_dir, subject_id):
    path = os.path.join(dataset_dir, 'RT_GENE_train_' + '{}.mat'.format(subject_id))
    out_dir = r'D:\RT-Gene_Dataset\mutiprocess_data\\'
    data = h5py.File(path)
    L_images = data['train/imagesL']
    R_images = data['train/imagesR']
    headpose = data['train/headposes']
    gazes = data['train/gazes']

    L_images = np.array(L_images)
    R_images = np.array(R_images)
    headpose = np.array(headpose)
    gazes = np.array(gazes)
    sio.savemat(out_dir+'RT_GENE_train_' + '{}.mat'.format(subject_id), {'imagesL':L_images, 'imagesR':R_images, 'headposes': headpose, 'gazes': gazes})
    print('data {} saved to {}'.format(subject_id, out_dir))


if __name__ == '__main__':
    path = r'D:\RT-Gene_Dataset\augment\\'
    subjects_test_threefold =['s000', 's001', 's002', 's003', 's004', 's007', 's008', 's009', 's010', 's005', 's006', 's011', 's012', 's013', 's014', 's015', 's016']

    subject_ids = ['%2s'%subject for subject in zip(subjects_test_threefold)]
    # subject_id = r'RT_GENE_test_s009'
    for subject in subject_ids:
        preprocess(dataset_dir=path, subject_id=subject)