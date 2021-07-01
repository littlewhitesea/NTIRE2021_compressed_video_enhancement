from __future__ import print_function
import argparse
from math import log10
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_test_set
import time
import torch.backends.cudnn as cudnn
import cv2
import math
import sys
import datetime
from utils import Logger
import numpy as np
import torchvision.utils as vutils
from arch import DUNet
import time
import os

parser = argparse.ArgumentParser(description='PyTorch DUVE Example')
parser.add_argument('--scale', type=int, default=1, help="super resolution upscale factor")
parser.add_argument('--in_channel', type=int, default=3, help="the channel number of input image")
parser.add_argument('--n_feature', type=int, default=64, help="the channel number of feature map")
parser.add_argument('--n_block1', type=int, default=6, help="the block number of RCAB in Unet1")
parser.add_argument('--n_block2', type=int, default=10, help="the block number of RCAB in Unet2")
parser.add_argument('--testbatchsize', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=123')
parser.add_argument('--gpus', default='4', type=str, help='gpu ids (default: 0)')
parser.add_argument('--cuda', default=True, type=bool)
################################################################################
######### Please change the 'test_dir' with your path of the test_fixed-rate_release.zip.
######### Then, you can run this code with 'python test.py'.
# e.g.: /media/data4/wh/Track3_release_test_data/001/001.png
# e.g.: /media/data4/wh/Track3_release_test_data/010/600.png
################################################################################
parser.add_argument('--test_dir', type=str, default='/media/data4/wh/Track3_release_test_data/')
parser.add_argument('--save_test_log', type=str, default='./result/log')
parser.add_argument('--pretrain', type=str, default='../pretrained_model/weight/X1_10L_64_epoch_206.pth')
parser.add_argument('--image_out', type=str, default='../out/')
opt = parser.parse_args()

systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus

print(opt)


def main():
    sys.stdout = Logger(os.path.join(opt.save_test_log, 'test_' + systime + '.txt'))
    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
        use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = False
        torch.cuda.manual_seed(opt.seed)
    pin_memory = True if use_gpu else False
    n_c = opt.n_feature
    n_b1 = opt.n_block1
    n_b2 = opt.n_block2
    dunet = DUNet(opt.in_channel, n_c, n_b1, n_b2)  # initial filter generate network
    print(dunet)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in dunet.parameters()) * 4 / 1048576))
    # Model size
    # parameters = Model size / 4
    print('===> {}L model has been initialized'.format(n_b1 + n_b2))
    dunet = torch.nn.DataParallel(dunet)
    print('===> load pretrained model')
    if os.path.isfile(opt.pretrain):
        dunet.load_state_dict(torch.load(opt.pretrain, map_location=lambda storage, loc: storage))
        print('===> pretrained model is load')
    else:
        raise Exception('pretrain model is not exists')
    if use_gpu:
        dunet = dunet.cuda()

    ############## real data testing (only include LQ image)
    scene_list = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']  # the name of test file

    for scene_name in scene_list:
        test_set = get_test_set(opt.test_dir, opt.scale, scene_name)
        test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testbatchsize, shuffle=False,
                                 pin_memory=pin_memory, drop_last=False)
        print('===> DataLoading Finished')
        real_data_test(test_loader, dunet, scene_name)


def test_ensamble(LH, op):
    # print(type(LH))
    npLH = LH.cpu().numpy()
    if op == 'v':
        tfLH = npLH[:, :, :, :, ::-1].copy()
    elif op == 'h':
        tfLH = npLH[:, :, :, ::-1, :].copy()
    elif op == 't':
        tfLH = npLH.transpose((0, 1, 2, 4, 3)).copy()

    en_LH = torch.Tensor(tfLH).cuda()
    return en_LH


def test_ensamble_2(LH, op):
    npLH = LH.cpu().numpy()
    if op == 'v':
        tfLH = npLH[:, :, :, ::-1].copy()
    elif op == 'h':
        tfLH = npLH[:, :, ::-1, :].copy()
    elif op == 't':
        tfLH = npLH.transpose((0, 1, 3, 2)).copy()

    en_LH = torch.Tensor(tfLH).cuda()
    return en_LH


def real_data_test(test_loader, dunet, scene_name):
    train_mode = False
    dunet.eval()
    average_time = 0

    flag_num = 0

    for image_num, data in enumerate(test_loader):

        flag_num += 1
        x_input = data
        with torch.no_grad():

            ########## test with ensemble strategy ###########
            x_input = Variable(x_input).cuda()
            t0 = time.time()
            if True:
                x_input_list = [x_input]
                for tf in ['v', 'h', 't']:
                    x_input_list.extend([test_ensamble(t, tf) for t in x_input_list])
                prediction_list = [dunet(aug) for aug in x_input_list]
                for i in range(len(prediction_list)):
                    if i > 3:
                        prediction_list[i] = test_ensamble_2(prediction_list[i], 't')
                    if i % 4 > 1:
                        prediction_list[i] = test_ensamble_2(prediction_list[i], 'h')
                    if (i % 4) % 2 == 1:
                        prediction_list[i] = test_ensamble_2(prediction_list[i], 'v')
                prediction_cat = torch.cat(prediction_list, dim=0)
                prediction = prediction_cat.mean(dim=0, keepdim=True)

            ######### test without ensemble stategy ##########
            # x_input = Variable(x_input).cuda()
            # t0 = time.time()
            # prediction = dunet(x_input)

        torch.cuda.synchronize()
        t1 = time.time()
        print("===> Timer: %.4f sec." % (t1 - t0))
        prediction = prediction.unsqueeze(2)
        prediction = prediction.squeeze(0).permute(1, 2, 3, 0)  # [T,H,W,C]
        prediction = prediction.cpu().numpy()[:, :, :, ::-1]  # tensor -> numpy, rgb -> bgr
        save_img(prediction[0], scene_name, image_num)
        average_time += (t1 - t0)

    print("time: {:.4f}".format(average_time / flag_num))


def save_img(prediction, scene_name, image_num):
    # save_dir = os.path.join(opt.image_out, systime)
    save_dir = os.path.join(opt.image_out, scene_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_dir = os.path.join(save_dir, '{:03}'.format(image_num + 1) + '.png')
    cv2.imwrite(image_dir, prediction * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    main()
