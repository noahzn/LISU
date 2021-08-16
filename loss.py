"""
@FileName: loss.py
@Time    : 4/29/2020
@Author  : Ning Zhang
@GitHub: https://github.com/noahzn
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


class LossRetinex:
    def __init__(self):
        super(LossRetinex, self).__init__()


    def gradient(self, input_tensor, direction):
        smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = smooth_kernel_x
        elif direction == "y":
            kernel = smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, r, i):
        r = 0.299 * r[:, 0, :, :] + 0.587 * r[:, 1, :, :] + 0.114 * r[:, 2, :, :]
        r = torch.unsqueeze(r, dim=1)
        return torch.mean(self.gradient(i, "x") * torch.exp(-10 * self.ave_gradient(r, "x")) +
                          self.gradient(i, "y") * torch.exp(-10 * self.ave_gradient(r, "y")))

    def smooth_r(self, r):
        r = 0.299 * r[:, 0, :, :] + 0.587 * r[:, 1, :, :] + 0.114 * r[:, 2, :, :]
        r = torch.unsqueeze(r, dim=1)
        return torch.mean(self.ave_gradient(r, "x") + self.ave_gradient(r, "y"))


    def smooth_i(self, d, i):
        # d = torch.unsqueeze(d, dim=1)
        return torch.mean(self.gradient(i, "x") * torch.exp(-10 * self.ave_gradient(d, "x")) +
                          self.gradient(i, "y") * torch.exp(-10 * self.ave_gradient(d, "y")))

    def recon(self, r, i, s):
        return F.l1_loss(r * i, s)

    def init_illumination_loss(self, R, I):
        km = torch.mean(R, dim=1)


        return F.l1_loss(km, I)

    def max_rgb_loss(self, image, illumination):
        n, c, h, w = image.size()
        max_rgb, _ = torch.max(image, 1)
        max_rgb = max_rgb.unsqueeze(1)

        # return F.l1_loss(illumination, max_rgb)
        return torch.norm(illumination-max_rgb, 1)/(n*c*h*w)

class SoftHistogram(torch.nn.Module):
    def __init__(self, bins=255, min=0, max=255, sigma=3*25):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = torch.nn.Parameter(self.centers, requires_grad=False)

    def forward(self, x):
        bn, c = x.size()
        x = torch.unsqueeze(x, 1) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta

        x = x.sum(dim=1)
        x = x / x.sum(dim=1).unsqueeze(1)  # normalization
        x = x.view(bn, 640, 480)

        return x





def enhance_loss_l1(i_hat):
    loss = F.l1_loss(torch.ones_like(i_hat), i_hat)

    return torch.mean(loss)


def equal_reflectance(r_low, r_high):

    return F.l1_loss(r_low, r_high.detach())


class LijunLoss:
    def __init__(self, i, i_hat):
        self.ori_img_block = F.avg_pool2d(i, kernel_size=4, stride=4)
        self.new_img_block = F.avg_pool2d(i_hat, kernel_size=4, stride=4)
        # [8, 1, 60, 80]

        self.eloss = self.enhance_loss()
        self.bloss = self.block_loss()

        # print(self.eloss.data, self.bloss.data)

    def block_contrast(self, img, direction):
        if direction == 'right':
            kernel = [0, 1, -1]
            imgpad = F.pad(img, (1, 1, 0, 0, 0, 0, 0, 0), mode='constant', value=0)
            kernel = torch.FloatTensor([kernel]).view(1, 1, 1, 3).cuda()
        elif direction == 'left':
            kernel = [-1, 1, 0]
            imgpad = F.pad(img, (1, 1, 0, 0, 0, 0, 0, 0), mode='constant', value=0)
            kernel = torch.FloatTensor([kernel]).view(1, 1, 1, 3).cuda()
        elif direction == 'up':
            kernel = [[-1], [1], [0]]
            imgpad = F.pad(img, (0, 0, 1, 1, 0, 0, 0, 0), mode='constant', value=0)
            kernel = torch.FloatTensor([kernel]).view(1, 1, 3, 1).cuda()

        elif direction == 'down':
            kernel = [[0], [1], [-1]]
            imgpad = F.pad(img, (0, 0, 1, 1, 0, 0, 0, 0), mode='constant', value=0)
            kernel = torch.FloatTensor([kernel]).view(1, 1, 3, 1).cuda()

        return F.conv2d(imgpad, weight=kernel, stride=1, padding=0)

    def block_loss(self):
        ori_r = self.block_contrast(self.ori_img_block, direction='right')
        new_r = self.block_contrast(self.new_img_block, direction='right')
        s_r = self.block_smooth(ori_r, new_r)

        ori_l = self.block_contrast(self.ori_img_block, direction='left')
        new_l = self.block_contrast(self.new_img_block, direction='left')
        s_l = self.block_smooth(ori_l, new_l)

        ori_u = self.block_contrast(self.ori_img_block, direction='up')
        new_u = self.block_contrast(self.new_img_block, direction='up')
        s_u = self.block_smooth(ori_u, new_u)

        ori_d = self.block_contrast(self.ori_img_block, direction='down')
        new_d = self.block_contrast(self.new_img_block, direction='down')
        s_d = self.block_smooth(ori_d, new_d)

        return torch.sum(0.5*(s_l+s_r+s_u+s_d))

    def block_smooth(self, ori, new):
        return (ori - new) ** 2

    def enhance_loss(self):
        loss = torch.sign(self.new_img_block - 0.9) * (self.new_img_block - self.ori_img_block)
        # loss2 = F.l1_loss(i, i_hat)
        return torch.sum(loss)

    def get_loss(self):
        return 0.1*self.eloss + 10*self.bloss


def ce_loss(x, target):
    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    class_weights = torch.FloatTensor(weights).cuda()
    ce = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)
    return ce(x, target)


def mse_loss(x, target):
    mse = torch.nn.MSELoss().cuda()
    return mse(x, target)


def l1_loss(x, target):
    return F.l1_loss(x, target)

def margin_ranking_loss(r, i):
    r = 0.299 * r[:, 0, :, :] + 0.587 * r[:, 1, :, :] + 0.114 * r[:, 2, :, :]
    # r = torch.unsqueeze(r, dim=1)
    margin_loss = torch.nn.MarginRankingLoss().cuda()
    target = torch.full_like(i, 1).cuda()

    return margin_loss(r, i, target)

class loss_huber(torch.nn.Module):
    def __init__(self):
        super(loss_huber,self).__init__()

    def forward(self, pred, truth):
        c = pred.shape[1] #通道
        h = pred.shape[2] #高
        w = pred.shape[3] #宽
        pred = pred.view(-1, c * h * w)
        truth = truth.view(-1, c * h * w)
        # 根据当前batch所有像素计算阈值
        t = 0.2 * torch.max(torch.abs(pred - truth))
        # 计算L1范数
        l1 = torch.mean(torch.mean(torch.abs(pred - truth), 1), 0)
        # 计算论文中的L2
        l2 = torch.mean(torch.mean(((pred - truth)**2 + t**2) / t / 2, 1), 0)

        if l1 > t:
            return l2
        else:
            return l1


def gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    gradient_h = F.pad(gradient_h, [0, 0, 1, 1], 'replicate')
    gradient_w = F.pad(gradient_w, [1, 1, 0, 0], 'replicate')
    gradient2_h = (img[:,:,4:,:]-img[:,:,:height-4,:]).abs()
    gradient2_w = (img[:, :, :, 4:] - img[:, :, :, :width-4]).abs()
    gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], 'replicate')
    gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], 'replicate')
    return gradient_h*gradient2_h, gradient_w*gradient2_w

def illumination_smooth_loss(image, illumination):
    # Gaussian Kernel Initialization
    n, c, h, w = image.size()
    g_kernel_size = 5
    g_padding = 2
    sigma = 3
    kx = cv2.getGaussianKernel(g_kernel_size, sigma)
    ky = cv2.getGaussianKernel(g_kernel_size, sigma)
    gaussian_kernel = np.multiply(kx, np.transpose(ky))
    gaussian_kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0).to('cuda')

    gray_tensor = 0.299*image[0,0,:,:] + 0.587*image[0,1,:,:] + 0.114*image[0,2,:,:]
    max_rgb, _ = torch.max(image, 1)
    max_rgb = max_rgb.unsqueeze(1)
    gradient_gray_h, gradient_gray_w = gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
    gradient_illu_h, gradient_illu_w = gradient(illumination)
    weight_h = 1/(F.conv2d(gradient_gray_h, weight=gaussian_kernel, padding=g_padding)+0.0001)
    weight_w = 1/(F.conv2d(gradient_gray_w, weight=gaussian_kernel, padding=g_padding)+0.0001)
    weight_h.detach()
    weight_w.detach()
    loss_h = weight_h * gradient_illu_h
    loss_w = weight_w * gradient_illu_w
    max_rgb.detach()
    return (loss_h.sum() + loss_w.sum() + 0.5*torch.norm(illumination-max_rgb, 1))/(n*c*h*w)
    # return torch.mean(loss_h.sum() + loss_w.sum()) + F.l1_loss(illumination, max_rgb)


class STDLoss(torch.nn.Module):
    def __init__(self):
        super(STDLoss, self).__init__()

    def forward(self, r, i):
        r_unfold = torch.nn.functional.unfold(r, kernel_size=4, dilation=1, stride=4)
        n, c_kh_kw, l = r_unfold.size()
        r_unfold = r_unfold.permute(0, 2, 1).view(n, l, -1, 4, 4)
        # print(r_unfold.size())
        x2 = torch.std(r_unfold, dim=(3, 4))
        loss = x2.mean()
        # if torch.isnan(loss):
        #     print(r_unfold)
        return loss

def gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    #print('x_data :',x_data.shape)

    #x_data = np.expand_dims(x_data, axis=-1)
    #x_data = np.expand_dims(x_data, axis=-1)
    x_data = torch.Tensor(x_data).unsqueeze(0).unsqueeze(0)

    #print('x_data :', y_data)

    y_data =  torch.Tensor(y_data).unsqueeze(0).unsqueeze(0)

    #y_data = np.expand_dims(y_data, axis=-1)
    #y_data = np.expand_dims(y_data, axis=-1)
    #print('x_data2 :', x_data.shape)
    #x = tf.constant(x_data, dtype=tf.float32)
    #y = tf.constant(y_data, dtype=tf.float32)

    #g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    g = torch.exp(-((x_data **2 + y_data **2)/(2.0 * sigma ** 2)))
    return g / torch.sum(g)


def pt_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    #img1 = torch.Tensor(img1)
    #img2 = torch.Tensor(img2)
    window = gauss(size, sigma) # window shape [size, size]
    K1 = torch.Tensor([0.01]).cuda()
    K2 = torch.Tensor([0.03]).cuda()
    L = torch.Tensor([1]).cuda()  # depth of image (255 in case the image has a differnt scale)
    C1 = torch.pow(K1*L,2)
    C2 = torch.pow(K2*L,2)
    #mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    #mu1  = nn.Parameter(data=window, requires_grad=False)
    weight = torch.nn.Parameter(data=window, requires_grad=False).cuda()
    mu1 = F.conv2d(img1,weight).cuda()
    #mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu2 = F.conv2d(img2,weight).cuda()
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    #sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID')
    sigma1_sq = F.conv2d(img1*img1,weight).cuda()- mu1_sq
    #sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma2_sq = F.conv2d(img2*img2,weight).cuda()- mu2_sq
    #sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    sigma12 =  F.conv2d(img1*img2,weight).cuda()- mu1_mu2

    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = torch.mean(value)
    #print('pt ' ,value)
    return value


def pt_ssim_loss(output_r, input_high_r):
    output_r_1 = output_r[:,0:1,:,:]  # R
    input_high_r_1 = input_high_r[:,0:1,:,:]
    ssim_r_1 = pt_ssim(output_r_1, input_high_r_1)
    #print('pt r_1', ssim_r_1)
    output_r_2 = output_r[:,1:2,:,:]   #G
    input_high_r_2 = input_high_r[:,1:2,:,:]
    ssim_r_2 = pt_ssim(output_r_2, input_high_r_2)
    #print('pt r_1', ssim_r_2)
    output_r_3 = output_r[:,2:3,:,:]  #B
    input_high_r_3 = input_high_r[:,2:3,:,:]
    ssim_r_3 = pt_ssim(output_r_3, input_high_r_3)
    #print('pt r_1', ssim_r_3)
    ssim_r = (ssim_r_1 + ssim_r_2 + ssim_r_3)/3.0
    loss_ssim1 = 1-ssim_r
    # loss_ssim1 = ssim_r
    # print('pt ssim loss :', loss_ssim1)
    return loss_ssim1


class RGB2Gray(torch.nn.Module):
    def __init__(self):
        super(RGB2Gray, self).__init__()
        _kernel = [0.299, 0.587, 0.114]#[0.2125, 0.7154, 0.0721]
        _kernel = torch.tensor(_kernel).view(1, 3, 1, 1)
        self.weight = _kernel.cuda()

    def forward(self, x):
        #print('weight pos:',self.weight.device)#cpu
        gray =  F.conv2d(x, self.weight)
        return gray


def gradient2(input_tensor, direction):
    # input_tensor = torch.FloatTensor(input_tensor)
    # print('input_tensor shape',input_tensor.shape)
    a = input_tensor.shape[0]

    b = torch.zeros(input_tensor.shape[2], 1)
    b = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], 1)
    b = b.cuda()

    # b = b.unsqueeze(0).unsqueeze(0)
    # print('b shape:',b.shape)
    # print('B',a)
    input_tensor = torch.cat((input_tensor, b), 3)

    # print('after cat input_tensor', input_tensor.shape)
    a = torch.zeros(1, input_tensor.shape[3])
    a = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], 1, input_tensor.shape[3])
    a = a.cuda()

    # a = a.unsqueeze(0).unsqueeze(0)
    # print('a', a.shape)
    input_tensor = torch.cat((input_tensor, a), 2)

    # print('input_tensor 2', input_tensor.shape)
    c = [[0, 0], [-1, 1]]
    c = torch.FloatTensor(c)
    c = c.cuda()

    # nn.init.constant(a,[[0, 0], [-1, 1]])
    # smooth_kernel_x = torch.reshape(nn.init.constant([[0, 0], [-1, 1]], torch.float32), (2, 2, 1))#torch.reshape()
    smooth_kernel_x = torch.reshape(c, (1, 1, 2, 2))  # unsqueeze()
    smooth_kernel_y = smooth_kernel_x.permute([0, 1, 3, 2])

    # print('gradient_orig:', smooth_kernel_y)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    weight = torch.nn.Parameter(data=kernel, requires_grad=False)
    gradient_orig = torch.abs(F.conv2d(input_tensor, weight, stride=1, padding=0))

    # c = gradient_orig
    # print('c shape',c.shape)
    # c = c.permute([0,2,3,1]).cpu().detach().numpy()
    # print('c shape',c[0])
    # cv2.imwrite('./gradient.jpg',c[0]*255)

    grad_min = torch.min(gradient_orig)  # https://blog.csdn.net/devil_son1234/article/details/105542067  torch.min
    grad_max = torch.max(gradient_orig)  # torch.max()
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))  # torch.div

    # print('pt grad norm',grad_norm)

    # c.weight = kernel
    # gradient_orig = c(input_tensor)
    # print('pt conv:',gradient_orig)
    # print('pt conv shape:', gradient_orig.shape)
    # print('smooth_kernel_x:',smooth_kernel_x)
    # print('smooth2:', tf.constant([[0, 0], [-1, 1]]))
    # smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])#torch.transpose()

    return grad_norm

def pt_grad_loss(input_r_low, input_r_high):
    gray = RGB2Gray().cuda()
    input_r_low_gray = gray(input_r_low)
    input_r_high_gray = gray(input_r_high)
    x_loss = torch.pow(gradient2(input_r_low_gray, 'x') - gradient2(input_r_high_gray, 'x'),2)
    y_loss = torch.pow(gradient2(input_r_low_gray, 'y') - gradient2(input_r_high_gray, 'y'),2)
    grad_loss_all = torch.mean(x_loss + y_loss)
    return grad_loss_all