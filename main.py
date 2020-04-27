import torch
from torchcontrib.optim import SWA
from Shallow_Model import DeepEyeNet
from dataloader import get_loader
from eyediap_loader import loader
import os
import shutil
import torch.backends.cudnn as cudnn
from torch.optim import Adam, lr_scheduler, SGD, ASGD, Adadelta
import time
import numpy as np
import cv2
from torch import nn, cuda, utils
import math

count_test = 0
count = 0
doLoad = False
doTest = False

def main():


    # GPU available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print('__Number CUDA Devices:', cuda.device_count())
    print('Active CUDA Device: GPU', cuda.current_device())

    global args, best_prec1, weight_decay, momentum

    data_path = r'D:\PycharmProjects\LBP-Gaze-Estimation\eye_detector\mpii_final_augmented\\'
    # data_path = r'D:\multi view dataset\\'
    # data_path = r'D:\PycharmProjects\shallow_network\eyeDiap_CS_S.h5'
    # data_path = '../eye-tracking/mpii_data/'



    base_lr = 0.0001
    epochs = 100
    workers = 0
    momentum = 0.9
    weight_decay = 1e-3
 #   best_prec1 = 1e20
    k_fold = 3


    model = DeepEyeNet()
    model = nn.DataParallel(model, device_ids=[0])

    cudnn.benchmark = True
    model.cuda()

    batch_size = cuda.device_count() * 256 # Change if out of cuda memory

    #opt = SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    # opt = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
    opt = Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    #opt = SGD(model.parameters(), lr=base_lr, weight_decay=weight_decay, momentum=momentum)
    # adadelt_opt = Adadelta()

    criterion = nn.MSELoss()

    subjects_test_threefold = [
        ['p03', 'p06', 'p01'],
        ['p02', 'p04', 'p05'],
        ['p07', 'p00', 'p08', 'p09', 'p10', 'p11']
    ]
    subjects_train_threefold = [
       ['p10', 'p02', 'p00', 'p08', 'p10', 'p14', 'p05', 'p04', 'p07', 'p09', 'p11', 'p13', 'p12'],
        ['p10', 'p03', 'p01', 'p00', 'p08', 'p14', 'p06', 'p11', 'p06', 'p12', 'p13', 'p09', 'p07'],
        ['p02', 'p03', 'p04', 'p07', 'p01', 'p05', 'p06', 'p12', 'p13']
    ]
    # val_set = ['p10', 'p02']

    if doLoad:
        saved = load_checkpoint()
        if saved:
            print('Loading checkpoint for epoch %05d with loss %.5f (which is L2 = mean of squares)...' % (saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            try:
                model.module.load_state_dict(state)
            except:
                model.load_state_dict(state)
            epoch = saved['epoch']
            best_prec1 = saved['best_prec1']
        else:
            print('Warning: Could not read checkpoint!')

    if doTest:
        state = model.state_dict()
        test_file = ['p02', 'p04', 'p05']
        UT_test = ['s20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29']
        print('testing is about to begin....')
        dataTest = get_loader(data_path, test_file)
        # dataTest = loader(data_path)
        test_loader = torch.utils.data.DataLoader(
            dataTest,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)
            # train(test_loader, model, criterion, adam_opt, epoch, k=0)
        validate(test_loader, model, criterion, epoch=0)
        return

    # state = model.state_dict()
    # for k in range(0, k_fold):
    k = 0
    best_prec1 = 1e20
    epoch = 0
    print('K fold Number : {}'.format(k))
    # initialize weights
    # model.load_state_dict(state)

    # if k == 2:
    #   opt = Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    data = get_loader(data_path, subjects_train_threefold[k], d_type='train')
    # data = loader(data_path)

    # val_data = get_loader(data_path, val_set)
    train_size = int(0.85 * len(data))
    val_size = int(len(data) - train_size)
    # test_size = int(len(data) - (train_size + val_size))

    train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])

    train_loader = utils.data.DataLoader(
        train_data,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=True)

    val_loader = utils.data.DataLoader(
        val_data,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    # test_loader = torch.utils.data.DataLoader(
    #     test_data,
    #     batch_size=batch_size, shuffle=True,
    #     num_workers=workers, pin_memory=True)
    #   if k == 1:
    #        base_lr = 0.001
    #         opt = SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)

    # elif k == 0:
    #     base_lr = 0.01
    #     opt = adam_opt
    #      else:
    #           opt = adam_opt

    scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3, verbose=True)
    # for epoch in range(0, epoch):
    # adjust_learning_rate(opt, epoch, base_lr)

    for epoch in range(0, epochs):
        # adjust_learning_rate(opt, epoch, base_lr)
        # train for one epoch
        train(train_loader, model, criterion, opt, epoch, k)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)
        scheduler.step(prec1)

        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        save_checkpoint(k, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
        print('learning rate is {}...'.format(base_lr))

def train(train_loader, model, criterion, optimizer, epoch, k):
    global count
    # logger.info('Train {}'.format(epoch))
    # clipper = UnitNormClipper()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    angle_error_meter = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    # list_Y = []

    for i, (imEyeL, imEyeR, headpose, gaze) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        imEyeL = imEyeL.cuda(async=True)
        imEyeR = imEyeR.cuda(async=True)
        # imFace = imFace.cuda(async=True)
        headpose = headpose.cuda(async=True)
        # landmark = landmark.cuda(async=True)
        gaze = gaze.cuda(async=True)

        imEyeL = torch.autograd.Variable(imEyeL, requires_grad=True)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad=True)
        # imFace = torch.autograd.Variable(imFace, requires_grad=True)
        headpose = torch.autograd.Variable(headpose, requires_grad=True)
        # landmark = torch.autograd.Variable(landmark, requires_grad=True)
        gaze = torch.autograd.Variable(gaze, requires_grad=True)


        # compute output
        optimizer.zero_grad()

        output = model(imEyeL, imEyeR, headpose)

        loss = criterion(output, gaze)

        angle_error = accuracy_angle_2(gaze, output)

        losses.update(loss.item(), imEyeL.size(0))
        angle_error_meter.update(angle_error.item(), imEyeL.size(0))


        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # if epoch % clipper.frequency == 0:
            # model.apply(clipper)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        count = count + 1

        print('K %d' %(k), 'Epoch (train): [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Angle_Loss {angle_error_meter.val: .4f} ({angle_error_meter.avg:.4f})'.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, angle_error_meter=angle_error_meter))
    # optimizer.swap_swa_sgd()

def validate(val_loader, model, criterion, epoch):
    # logger.info('Test {}'.format(epoch))

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()
    angle_error_Loss = AverageMeter()

    # switch to evaluate mode
    end = time.time()

    oIndex = 0
    for i, (imEyeL, imEyeR, headpose, gaze) in enumerate(val_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        imEyeL = imEyeL.cuda(async=True)
        imEyeR = imEyeR.cuda(async=True)
        # imFace = imFace.cuda(async=True)
        headpose = headpose.cuda(async=True)
        # landmark = landmark.cuda(async=True)
        gaze = gaze.cuda(async=True)

        imEyeL = torch.autograd.Variable(imEyeL)
        imEyeR = torch.autograd.Variable(imEyeR)
        # imFace = torch.autograd.Variable(imFace)
        headpose = torch.autograd.Variable(headpose)
        # landmark = torch.autograd.Variable(landmark)
        gaze = torch.autograd.Variable(gaze)

        with torch.no_grad():
            # compute output
            output = model(imEyeL, imEyeR, headpose)

        loss = criterion(output, gaze)
        angle_error = accuracy_angle_2(gaze, output)


        # lossLin = output - gaze
        # lossLin = torch.mul(lossLin,lossLin)
        # lossLin = torch.sum(lossLin,1)
        # lossLin = torch.mean(torch.sqrt(lossLin))

        losses.update(loss.item(), imEyeL.size(0))
        # lossesLin.update(lossLin.item(), imEyeL.size(0))
        angle_error_Loss.update(angle_error.item(), imEyeL.size(0))

        # logger.info('Epoch {} Loss {:.4f} AngleError {:.2f}'.format(
        #     epoch, loss.avg, angle_error_meter.avg))


        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        print('Epoch (val): [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'angle_error_Loss {angle_error_Loss.val: .4f} ({angle_error_Loss.avg: .4f})'.format(
                epoch, i, len(val_loader), batch_time=batch_time,
                loss=losses, angle_error_Loss=angle_error_Loss))
        # elapsed = time.time() - end
        # logger.info('Elapsed {:.2f}'.format(elapsed))
        # writer.add_scalar('val/loss', lossesLin.avg, epoch)
        # writer.add_scalar('val/angle_loss', angle_error_Loss.avg, epoch)
        #
        # writer.export_scalars_to_json("./all_scalars.json")
        # writer.close()

    return angle_error_Loss.avg

def convert_to_unit_vector(angles):
    x = -(torch.cos(angles[:, 0]) * torch.sin(angles[:, 1]))
    y = - (torch.sin(angles[:, 0]))
    z = -(torch.cos(angles[:, 1]) * torch.cos(angles[:, 1]))
    norm = torch.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z

def accuracy_angle_2(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    angle_accuracy = np.zeros(len(y_true))
    for i in range (0, len(y_true)):
        pred_x = -1*math.cos(y_pred[i, 0])*math.sin(y_pred[i, 1])
        pred_y = -1*math.sin(y_pred[i, 0])
        pred_z = -1*math.cos(y_pred[i, 0])*math.cos(y_pred[i, 1])
        pred_norm = math.sqrt(pred_x*pred_x + pred_y*pred_y + pred_z*pred_z)

        true_x = -1*math.cos(y_true[i, 0])*math.sin(y_true[i, 1])
        true_y = -1*math.sin(y_true[i, 0])
        true_z = -1*math.cos(y_true[i, 0])*math.cos(y_true[i, 1])
        true_norm = math.sqrt(true_x*true_x + true_y*true_y + true_z*true_z)

        angle_value = (pred_x*true_x + pred_y*true_y + pred_z*true_z) / (true_norm*pred_norm)
        np.clip(angle_value, -0.9999999999, 0.999999999)
        angle_degree = math.degrees(math.acos(angle_value))
        angle_accuracy[i] = angle_degree
    return angle_accuracy.mean()


def compute_angle_error(preds, labels):
    pred_x, pred_y, pred_z = convert_to_unit_vector(preds)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    return torch.acos(angles) * 180 / np.pi

# CHECKPOINTS_PATH = '../eye-tracking/shallow_network'
CHECKPOINTS_PATH = r'D:\PycharmProjects\shallow_network\saved 3 (good model)\\'
def load_checkpoint(filename='best_0_checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
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


def weights_init(m):
    if type(m) in [nn.Linear]:
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0)


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


def adjust_learning_rate(optimizer, epochs, base_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.01 ** (epochs//5))
    print('Learning Rate decreased to {}'.format(lr))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


# def visualize_eye_result(eye_image, est_gaze):
#     """Here, we take the original eye eye_image and overlay the estimated gaze."""
#     output_image = np.copy(eye_image)
#     h, w, c = eye_image.shape()
#     center_x = image_width / 2
#     center_y = image_height / 2
#
#     endpoint_x, endpoint_y = gaze_tools.get_endpoint(est_gaze[0], est_gaze[1], center_x, center_y, 50)
#
#     cv2.line(output_image, (int(center_x), int(center_y)), (int(endpoint_x), int(endpoint_y)), (255, 0, 0))
#     return output_image

class UnitNormClipper(object):

    def __init__(self, frequency=2):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.div_(torch.norm(w, 2, 1).expand_as(w))

if __name__ == '__main__':
    main()





