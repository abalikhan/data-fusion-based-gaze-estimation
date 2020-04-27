from torch import cuda, optim, nn, utils
import os
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

#load model
from vgg_face import Vgg_face_dag, VGG_gaze
from train import train, test
from data_loader import get_loader
from utils import save_checkpoint, load_checkpoint, load_vgg_weights

#loading and training flags
doLoad = False
doTest = False


def main():
    # GPU available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print('__Number CUDA Devices:', cuda.device_count())
    print('Active CUDA Device: GPU', cuda.current_device())

    global args, best_prec1, weight_decay, momentum

    data_path = '/home/miruware/gaze/dataset'

    base_lr = 0.0001
    epochs = 30
    workers = 0
    momentum = 0.9
    weight_decay = 1e-3
    best_prec1 = 1e20
    batch_size = 10

    #initialize model
    model = VGG_gaze()  #pre-traind VGG face model

    model = nn.DataParallel(model, device_ids=[0])

    cudnn.benchmark = True
    model.cuda()

    #optimizer
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, betas=(0.9, 0.999), eps=1e-08)

    criterion = nn.SmoothL1Loss()

    exp_lr_scheduler = lr_scheduler.StepLR(opt, step_size=7, gamma=0.1)

    num_persons = 15

    #load vgg weights and freeze all pre-trained parameters

    # weights_path = 'vgg_face_dag.pth'        # vgg face pre-trained weights path
    # vgg_face = load_vgg_weights(vgg_face, weights_path)

    # for param in vgg_face.parameters():
    #     param.requires_grad_(False)

#LeaveOneOut Scenario
    for idx in range(0, num_persons):
        best_prec1 = 1e20

        train_loader, test_loader = get_loader(data_path, idx, batch_size=batch_size, num_workers=workers)

        for epoch in range(1, epochs+1):

            #apply lr scheduler
            exp_lr_scheduler.step()

            #train the model
            train(train_loader, model, criterion=criterion,
                  optimizer=opt, epoch=epoch, k=idx)

            prec1 = test(test_loader, model, criterion, epoch)

            #remember the best prec and save checkpoint
            is_best = prec1 < best_prec1
            best_prec1 = min(prec1, best_prec1)
            save_checkpoint(idx, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
            print('learning rate is {}...'.format(base_lr))



if __name__ == '__main__':
    main()




