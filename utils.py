'''
    Important utilities
'''
import torch
import numpy as np
import os
import shutil

def convert_to_unit_vector(angles):
    x = - torch.cos(angles[:, 0]) * torch.sin(angles[:, 1])
    y = - torch.sin(angles[:, 0])
    z = - torch.cos(angles[:, 1]) * torch.cos(angles[:, 1])
    norm = torch.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_angle_error(preds, labels):
    pred_x, pred_y, pred_z = convert_to_unit_vector(preds)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    return torch.acos(angles) * 180 / np.pi


CHECKPOINTS_PATH = '~/gaze/models/'
def load_checkpoint(k):
    filename = os.path.join(CHECKPOINTS_PATH, 'best_%d_checkpoint.pth.tar'%k)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state

def save_checkpoint(k, state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)

    bestFilename = os.path.join(CHECKPOINTS_PATH, 'best_%d_' %(k) + filename)
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename)


def load_vgg_weights(model, weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model