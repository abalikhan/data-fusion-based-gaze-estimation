import time
import torch
import logging

from utils import AverageMeter, compute_angle_error

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

def train(train_loader, model, criterion, optimizer, epoch, k):


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    angle_error_meter = AverageMeter()

    # switch to train mode
    model.train()
    # gaze_model.train()

    end = time.time()


    for i, (imFace, gaze) in enumerate(train_loader):

        data_time.update(time.time() - end)

        imFace = imFace.cuda(async=True)
        gaze = gaze.cuda(async=True)


        imFace = torch.autograd.Variable(imFace, requires_grad=True)
        gaze = torch.autograd.Variable(gaze, requires_grad=True)


        # compute output
        model.zero_grad()
        # gaze_model.zero_grad()

        #Pass the image to VGG-Face and Gaze output model
        # features = feature_extractor(imFace)
        output = model(imFace)

        #compute loss
        loss = criterion(output, gaze)

        #computer angle error
        angle_error = compute_angle_error(gaze, output).mean()

        losses.update(loss.item(), imFace.size(0))
        angle_error_meter.update(angle_error.item(), imFace.size(0))


        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()



        print('Subject %d' %(k), 'Epoch (train): [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Angle_Loss {angle_error_meter.val: .4f} ({angle_error_meter.avg:.4f})'.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, angle_error_meter=angle_error_meter))
    return

def test(val_loader, model, criterion, epoch):

    model.eval()
    # gaze_model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    angle_error_Loss = AverageMeter()

    # switch to evaluate mode
    end = time.time()

    for i, (imFace, gaze) in enumerate(val_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        imFace = imFace.cuda(async=True)
        gaze = gaze.cuda(async=True)

        imFace = torch.autograd.Variable(imFace)
        gaze = torch.autograd.Variable(gaze)

        with torch.no_grad():
            # compute output
            output = model(imFace)
            # output = gaze_model(feat)

        loss = criterion(output, gaze)
        angle_error = compute_angle_error(gaze, output).mean()

        losses.update(loss.item(), imFace.size(0))
        angle_error_Loss.update(angle_error.item(), imFace.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        print('Epoch (val): [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'angle_error_Loss {angle_error_Loss.val: .4f} ({angle_error_Loss.avg: .4f})'.format(
                epoch, i, len(val_loader), batch_time=batch_time,
                loss=losses, angle_error_Loss=angle_error_Loss))


    return angle_error_Loss.avg