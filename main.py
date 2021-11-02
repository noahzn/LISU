"""
@FileName: main.py
@Time    : 7/16/2020
@Author  : Ning Zhang
@GitHub: https://github.com/noahzn
"""

import os
from tqdm import tqdm
import csv
import matplotlib
from tensorboardX import SummaryWriter

matplotlib.use('Agg')
from loss import *
from dataset import LLRGBD_real, LLRGBD_synthetic
from PIL import Image
import utils

import models
import argparse
import sys
import time

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(profile='full')
# torch.backends.cudnn.benchmark = True


def train_and_val(args, Decomp, opt_Decomp, Enhance, opt_Enhance, trainloader, valloader, best_pred=0.0):
    writer_name = '{}_{}'.format('LISU', str(time.strftime("%m-%d %H:%M:%S", time.localtime())))
    writer = SummaryWriter(os.path.join('runs', writer_name))
    step = 0

    opt_Enhance.zero_grad()
    opt_Decomp.zero_grad()

    loss_decomp = LossRetinex()

    for epoch in range(args.start_epoch, args.num_epochs):
        lr = utils.poly_lr_scheduler(opt_Decomp, 0.001, epoch, max_iter=args.num_epochs)
        lr = utils.poly_lr_scheduler(opt_Enhance, 0.001, epoch, max_iter=args.num_epochs)
        print('epoch: ', epoch, ' lr: ', lr, 'best: ', best_pred)
        loss_record = []
        iou_record = []

        # if epoch > 150:
        #     for p in Decomp.parameters():
        #         p.requires_grad = False
        # else:
        #     Decomp.train()

        Decomp.train()
        Enhance.train()
        tbar = tqdm(trainloader)

        for i, (image, image2, label, name) in enumerate(tbar):
            train_loss = 0.0
            image = image.cuda()
            label = label.cuda()
            image2 = image2.cuda()

            I, R = Decomp(image)
            I2, R2 = Decomp(image2)
            I_3 = torch.cat((I, I, I), dim=1)
            I2_3 = torch.cat((I2, I2, I2), dim=1)

            recon_low = loss_decomp.recon(R, I_3, image)
            recon_high = loss_decomp.recon(R2, I2_3, image2)
            recon_low_2 = loss_decomp.recon(R2, I_3, image)
            recon_high_2 = loss_decomp.recon(R, I2_3, image2)
            smooth_i_low = loss_decomp.smooth(image, I)
            smooth_i_high = loss_decomp.smooth(image2, I2)
            max_i_low = loss_decomp.max_rgb_loss(image, I)
            max_i_high = loss_decomp.max_rgb_loss(image2, I2)

            loss1 = recon_low + recon_high + 0.01 * recon_low_2 + 0.01 * recon_high_2 \
                    + 0.5 * smooth_i_low + 0.5 * smooth_i_high + \
                    0.1 * max_i_low + 0.1 * max_i_high

            if epoch > 650:
                opt_Decomp.zero_grad()
            else:
                opt_Decomp.zero_grad()
                loss1.backward()
                opt_Decomp.step()

            R_hat, smap = Enhance(torch.cat((I.detach(), R.detach()), dim=1))

            recon_r = torch.mean(torch.pow((R_hat - R2.detach()), 2))
            ssim_r = pt_ssim_loss(R_hat, R2.detach())
            grad_r = pt_grad_loss(R_hat, R2.detach())

            loss_r = recon_r + ssim_r + grad_r
            loss_s = ce_loss(smap, label)
            loss = loss_r + loss_s

            opt_Enhance.zero_grad()
            loss.backward()
            opt_Enhance.step()
            step += 1


            smap_oh = utils.reverse_one_hot(smap)
            _, _, iou, _, iu = utils.label_accuracy_score(label.data.cpu().numpy(), smap_oh.data.cpu().numpy(),
                                                    args.num_classes)

            iou_record.append(np.nanmean(iu))

            train_loss += loss1.item() + loss.item()

            writer.add_scalar('loss_step', train_loss, step)
            loss_record.append(train_loss)

            tbar.set_description('TrainLoss: {0:.3} mIoU: {1:.3} S: {2:.3}'.format(
                np.mean(loss_record), np.nanmean(iou_record), loss_r))

            if i % 600 == 0:
                I_pow = torch.pow(I_3, 0.1)
                I_e = I_pow * R_hat
                sout = utils.reverse_one_hot(smap)
                sout = sout[0, :, :]
                sout = utils.colorize(sout).numpy()
                sout = np.transpose(sout, (2, 0, 1))

                lbl = label[0, :, :]
                lbl = utils.colorize(lbl).numpy()
                lbl = np.transpose(lbl, (2, 0, 1))
                cat_image = np.concatenate(
                    [image[0, :, :, :].detach().cpu(), R[0, :, :, :].detach().cpu(), I_3[0, :, :, :].detach().cpu(),
                     R2[0, :, :, :].detach().cpu(), I2_3[0, :, :, :].detach().cpu(), R_hat[0, :, :, :].detach().cpu(),
                     I_e[0, :, :, :].detach().cpu(), lbl, sout], axis=2)
                cat_image = np.transpose(cat_image, (1, 2, 0))
                cat_image = np.clip(cat_image * 255.0, 0, 255.0).astype('uint8')

                im = Image.fromarray(cat_image)
                filepath = os.path.join('/path/to/folder', 'train_%d_%d.png' %
                                        (epoch, i))

                im.save(filepath[:-4] + '.jpg')

        tbar.close()
        loss_train_mean = np.mean(loss_record)
        iou_train = np.nanmean(iou_record)

        writer.add_scalar('iou_train', iou_train, epoch)
        writer.add_scalar('loss_epoch_train', float(loss_train_mean), epoch)


        if epoch % args.validation_step == 0:
            iou, loss_val, acc, pre_val, lbl_val, avgacc = val(args, Decomp, opt_Decomp, Enhance, opt_Enhance, trainloader, valloader)
            writer.add_scalar('iou_val', iou, epoch)
            writer.add_scalar('acc_val', acc, epoch)
            writer.add_scalar('loss_epoch_val', loss_val, epoch)

            if not os.path.isdir(args.save_model_path):
                os.makedirs(args.save_model_path)

            print('iou: {}, acc: {}, best: {}, new: {}'.format(iou, acc, best_pred, (iou + acc) / 2))

            if (iou + acc) / 2 > best_pred:
                best_pred = (iou + acc) / 2

                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_decomp': Decomp.state_dict(),
                    'optimizer_decomp': opt_Decomp.state_dict(),
                    'state_dict_enhance': Enhance.state_dict(),
                    'optimizer_enhance': opt_Enhance.state_dict(),
                    'best_pred': best_pred,
                }, is_best=True, filename='best.pth.tar')
            else:
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_decomp': Decomp.state_dict(),
                    'optimizer_decomp': opt_Decomp.state_dict(),
                    'state_dict_enhance': Enhance.state_dict(),
                    'optimizer_enhance': opt_Enhance.state_dict(),
                    'best_pred': best_pred,
                }, is_best=False, filename='{0}.pth.tar'.format(epoch))

            with open('/path/to/log.csv', 'a') as csvfile:
                wt = csv.DictWriter(csvfile, fieldnames=['iou', 'acc', 'avgacc', 'average'])

                wt.writeheader()
                wt.writerow({'iou': iou, 'acc': acc, 'avgacc': avgacc, 'average': (iou+acc)/2})


def val(args, Decomp, opt_Decomp, Enhance, opt_Enhance, trainloader, valloader, best_pred=0.0):
    print('start val!')

    with torch.no_grad():
        Decomp.eval()
        Enhance.eval()
        loss_decomp = LossRetinex()
        lbls = []
        preds = []
        loss_record = []
        iou_record = []

        tbar = tqdm(valloader)
        for i, (image, image2, label, name) in enumerate(tbar):
            image = image.cuda()
            label = label.cuda()
            image2 = image2.cuda()
            # label_sd1 = label_sd1.cuda()
            # label_sd2 = label_sd2.cuda()
            # label_sd3 = label_sd3.cuda()

            if args.multiple_GPUs:
                # output = gather(output, 0, dim=0)
                output = output[0]

                I, R = Decomp(image)
                I2, R2 = Decomp(image2)
                I_3 = torch.cat((I, I, I), dim=1)
                I2_3 = torch.cat((I2, I2, I2), dim=1)

                R_hat, smap = Enhance(torch.cat((I.detach(), R.detach()), dim=1))

                recon1 = loss_decomp.recon(R, I, image)
                smooth_i = illumination_smooth_loss(R, I)
                max_i = loss_decomp.max_rgb_loss(image, I)
                loss1 = recon1 + 0.05*smooth_i +0.02 * max_i
                loss_s = ce_loss(smap, label)

                loss = loss1 + loss_s
                loss_record.append(loss.item())

                smap_oh = utils.reverse_one_hot(smap)

                for l, p in zip(label.data.cpu().numpy(), smap_oh.data.cpu().numpy()):
                    lbls.append(l)
                    preds.append(p)

                if i % 5 == 0:
                    I_pow = torch.pow(I_3, 0.1)
                    I_e = I_pow * R_hat
                    sout = utils.reverse_one_hot(smap)
                    sout = sout[0, :, :]
                    sout = utils.colorize(sout).numpy()

                    sout = np.transpose(sout, (2, 0, 1))

                    lbl = label[0, :, :]
                    lbl = utils.colorize(lbl).numpy()
                    lbl = np.transpose(lbl, (2, 0, 1))
                    cat_image = np.concatenate([image[0, :, :, :].detach().cpu(), R[0, :, :, :].detach().cpu(),
                                                I_3[0, :, :, :].detach().cpu(), R2[0, :, :, :].detach().cpu(),
                                                I2_3[0, :, :, :].detach().cpu(),
                                                R_hat[0, :, :, :].detach().cpu(),
                                                I_e[0, :, :, :].detach().cpu(), lbl, sout], axis=2)
                    cat_image = np.transpose(cat_image, (1, 2, 0))
                    cat_image = np.clip(cat_image * 255.0, 0, 255.0).astype('uint8')

                    im = Image.fromarray(cat_image)
                    filepath = os.path.join('/path/to/save/folder', 'val_%d.png' % i)
                    im.save(filepath[:-4] + '.jpg')

        acc, acc_cls, iou, _, iu = utils.label_accuracy_score(lbls, preds, args.num_classes)

    return np.sum(np.nanmean(iu[1:])), np.mean(loss_record), acc, smap_oh, label, np.sum(np.nanmean(acc_cls[1:]))


def output(args, Net, Net2, valloader):
    print('start output!')

    with torch.no_grad():
        Net.eval()
        Net2.eval()

        tbar = tqdm(valloader)
        for i, (image, image2, label, name) in enumerate(tbar):
            image = image.cuda()
            label = label.cuda()
            image2 = image2.cuda()

            if args.multiple_GPUs:
                output = output[0]

            else:
                I, R = Net(image)
                I2, R2 = Net(image2)
                R_hat, smap = Net2(torch.cat((I.detach(), R.detach()), dim=1))

                sout = utils.reverse_one_hot(smap)
                sout = sout[0, :, :]
                sout = utils.colorize(sout).numpy()
                sout = np.transpose(sout, (2, 0, 1))

                lbl = label[0, :, :]
                lbl = utils.colorize(lbl).numpy()
                lbl = np.transpose(lbl, (2, 0, 1))

                I_3 = torch.cat((I, I, I), dim=1)
                I2_3 = torch.cat((I2, I2, I2), dim=1)
                I_pow = torch.pow(I_3, 0.3)
                I_e = I_pow * R_hat

                output_list = [I_e, I_3, I2_3, R, R_hat, sout, lbl, R2]
                output_list_name = ['enhanced', 'i', 'i2', 'r', 'r_hat', 'seg', 'lbl', 'r2']

                for x in range(len(output_list)):
                    if x == 5:
                        cat_image = np.concatenate([sout], axis=2)
                    elif x == 6:
                        cat_image = np.concatenate([lbl], axis=2)
                    else:
                        cat_image = np.concatenate([output_list[x][0, :, :, :].detach().cpu()], axis=2)
                    filepath = os.path.join('/path/to/output_lisujoint_t', '%s_%s.png' % (name[0], output_list_name[x]))

                    cat_image = np.transpose(cat_image, (1, 2, 0))
                    cat_image = np.clip(cat_image * 255.0, 0, 255.0).astype('uint8')

                    im = Image.fromarray(cat_image)
                    im.save(filepath[:-4] + '.png')


def evaluation(args, Net, Net2, valloader):
    print('start evaluation!')

    with torch.no_grad():
        Net.eval()
        Net2.eval()
        lbls = []
        preds = []

        tbar = tqdm(valloader)
        for i, (image, image2, label, name) in enumerate(tbar):
            image = image.cuda()

            if args.multiple_GPUs:
                output = output[0]

            else:
                I, R = Net(image)
                R_hat, smap = Net2(torch.cat((I.detach(), R.detach()), dim=1))

                smap_oh = utils.reverse_one_hot(smap)

                for l, p in zip(label.data.cpu().numpy(), smap_oh.data.cpu().numpy()):
                    lbls.append(l)
                    preds.append(p)

        acc, acc_cls, iou, _, iu = utils.label_accuracy_score(lbls, preds, args.num_classes)

        print(iu)
        print(np.sum(np.nanmean(iu[1:])))
        print(acc)
        print(acc_cls)
        print(np.sum(np.nanmean(acc_cls[1:])))


def main(argv):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train_and_val', help='train_and_val|output|evaluation')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train for')
    parser.add_argument('--data_path', type=str, default='/path/to/your/dataset/', help='path to your dataset')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--crop_height', type=int, default=576, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=768, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of images in each batch')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--print_freq', type=int, default=600, help='print freq')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=14, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU id used for training')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default='/path/to/LISU_LLRGBD_real_best.pth.tar', help='saved model')
    parser.add_argument('--save_model_path', type=str, default='./checkpoints', help='path to save trained model')
    parser.add_argument('--multiple-GPUs', default=False, help='train with multiple GPUs')
    args = parser.parse_args(argv)


    # gpu id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    import torch
    from torch.utils.data import DataLoader

    # random seed
    np.random.seed(args.seed)  # cpu vars
    torch.manual_seed(args.seed)  # cpu  vars
    torch.cuda.manual_seed_all(args.seed)

    Net = models.LISU_DECOMP().cuda()
    Net2 = models.LISU_JOINT().cuda()

    opt_Net = torch.optim.Adam(Net.parameters(), lr=0.001, betas=(0.95, 0.999))
    opt_Net2 = torch.optim.Adam(Net2.parameters(), lr=0.001, betas=(0.95, 0.999))

    if args.pretrained_model_path is not None:
        if not os.path.isfile(args.pretrained_model_path):
            raise RuntimeError("=> no pretrained model found at '{}'".format(args.pretrained_model_path))

        checkpoint = torch.load(args.pretrained_model_path)
        Net.load_state_dict(checkpoint['state_dict_decomp'])
        Net2.load_state_dict(checkpoint['state_dict_enhance'])
        args.start_epoch = checkpoint['epoch']
        opt_Net.load_state_dict(checkpoint['optimizer_decomp'])
        opt_Net2.load_state_dict(checkpoint['optimizer_enhance'])
        best_pred = checkpoint['best_pred']
    else:
        best_pred = 0.0


    trainset = LLRGBD_real(args, mode='train')
    #trainset = LLRGBD_synthetic(args, mode='train')
    valset = LLRGBD_real(args, mode='val')
    # valset = LLRGBD_synthetic(args, mode='val')
    trainloader = DataLoader(trainset, batch_size=12, shuffle=True, num_workers=16, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=16, drop_last=False)


    if args.mode == 'train_and_val':
        train_and_val(args, Net, opt_Net, Net2, opt_Net2, trainloader, valloader, best_pred=best_pred)

    elif args.mode == 'output':
        output(args, Net, Net2, valloader)

    elif args.mode == 'evaluation':
        evaluation(args, Net, Net2, valloader)


if __name__ == '__main__':
    main(sys.argv[1:])
