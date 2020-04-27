import numpy as np
import h5py
import os
import cv2
import scipy.io as sio

def data_processing(dataset_dir):

    # path = os.path.join(dataset_dir)
    out_path = r'D:\MPIIFaceGaze\MPIIFACE_Processed'

    FACES = []
    GAZES = []

    for i in range(0, 3):
        sub_id = ('p{:02}'.format(i))
        path = os.path.join(dataset_dir, '{}.h5'.format(sub_id))
        f = h5py.File(path, 'r')
        images = f['/Data/data'][:]
        gz = f['/Data/label'][:]

        if i == 1:
            continue
        idx = 0
        for (img, label) in zip(images, gz):

            face = img[[2, 1, 0], :, :].transpose((1, 2, 0))
            face = cv2.resize(face, (224, 224), cv2.INTER_CUBIC)
            face = face/255.

            FACES.append(face)
            GAZES.append(label)

            idx +=1

        images_array = np.asarray(FACES)
        images_array = images_array.reshape(3000, -1)
        labels_array = np.asarray(GAZES)
        sio.savemat(os.path.join(out_path, '{}.mat'.format(sub_id)), {'faces': images_array, 'labels': labels_array})
        print('data preprocessed successful!!')
        FACES.clear()
        GAZES.clear()


if __name__ == '__main__':
    data_processing(r'D:\MPIIFaceGaze\MPIIFaceGaze_normalizad')