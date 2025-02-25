from tkinter import Variable
import torch.nn as nn
import torch.nn.init as init
import math
from einops import rearrange
import numpy
from tkinter import Variable
import numpy as np
import math
import torch
import cv2
from torch import nn, fft, unsqueeze
import torch.nn.functional as F
from torch.distributions import kl, Independent, Normal
from torch.profiler import profile
import pywt
from fusion import SpaFre
# from loss_functions import DJSLoss



CE = torch.nn.BCELoss(reduction='sum')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



class DJSLoss(nn.Module):
    """Jensen Shannon Divergence loss"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, T: torch.Tensor, T_prime: torch.Tensor) -> float:
        """Estimator of the Jensen Shannon Divergence see paper equation (2)

        Args:
            T (torch.Tensor): Statistique network estimation from the marginal distribution P(x)P(z)
            T_prime (torch.Tensor): Statistique network estimation from the joint distribution P(xz)

        Returns:
            float: DJS estimation value
        """
        joint_expectation = (-F.softplus(-T)).mean()
        marginal_expectation = F.softplus(T_prime).mean()
        # print('joint_expectation:' ,joint_expectation)
        # print('marginal_expectation:',marginal_expectation)
        mutual_info = joint_expectation - marginal_expectation

        return -mutual_info
        

def Gaussian_filter(tensor, kernel_size, sigma=0):
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("Kernel size must be a positive odd integer greater than or equal to 3.")

    if sigma == 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    # convert tensor to numpy
    numpy_array = tensor.detach().cpu().numpy()

    # Gaussian_filter
    for i in range(numpy_array.shape[0]):  # iterate over the batch size
        for channel in range(numpy_array.shape[1]):  # iterate over channels
            image = numpy_array[i, channel, :, :]  # Retrieve current channel

            # Gaussian_blur
            blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

            # filtered result to the original array
            numpy_array[i, channel, :, :] = blurred_image

    # convert numpy to tensor
    blurred_tensor = torch.from_numpy(numpy_array)
    blur_lp = torch.tensor(blurred_tensor).to(device)
    # blur_hp = tensor - blur_lp

    return blur_lp





class Mutual_info_reg(nn.Module):
    def __init__(self, input_channels, channels,mid, latent_size = 4):
        super(Mutual_info_reg, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.channels = channels
        self.mid = mid
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.fc1_rgb3 = nn.Linear(channels * 1 * mid * mid, latent_size)
        self.fc2_rgb3 = nn.Linear(channels * 1 * mid * mid, latent_size)
        self.fc1_depth3 = nn.Linear(channels * 1 * mid * mid, latent_size)
        self.fc2_depth3 = nn.Linear(channels * 1 * mid * mid, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()


    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, fm, fp):
        fm = 255*self.layer3(self.leakyrelu(self.layer1(fm)))
        fp = 255*self.layer4(self.leakyrelu(self.layer2(fp)))

        # rgb_feat =
        fm = fm.view(-1, self.channels * 1 * self.mid * self.mid)
        fp = fp.view(-1, self.channels * 1 * self.mid * self.mid)
        mu_m = self.fc1_rgb3(fm)
        logvar_m = self.fc2_rgb3(fm)
        mu_p = self.fc1_depth3(fp)
        logvar_p = self.fc2_depth3(fp)

        mu_p = self.tanh(mu_p)
        mu_m = self.tanh(mu_m)
        logvar_p = self.tanh(logvar_p)
        logvar_m = self.tanh(logvar_m)
        z_rgb = self.reparametrize(mu_m, logvar_m)
        dist_rgb = Independent(Normal(loc=mu_m, scale=torch.exp(logvar_m)), 1)
        z_depth = self.reparametrize(mu_p, logvar_p)
        dist_depth = Independent(Normal(loc=mu_p, scale=torch.exp(logvar_p)), 1)

        # bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth))
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_depth_norm = torch.sigmoid(z_depth)
        ce_rgb_depth = CE(z_rgb_norm, z_depth_norm.detach())
        ce_depth_rgb = CE(z_depth_norm, z_rgb_norm.detach())
        # bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth))
        bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth)) + torch.mean(
            self.kl_divergence(dist_depth, dist_rgb))
        # bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth) + torch.mean(self.kl_divergence(dist_depth,dist_rgb)) )
        # latent_loss =  (- ce_rgb_depth -ce_depth_rgb )+bi_di_kld
        # latent_loss = ce_rgb_depth + ce_depth_rgb - bi_di_kld
        latent_loss =1 - bi_di_kld/(ce_rgb_depth+ce_depth_rgb)
        # latent_loss = torch.abs(cos_sim(z_rgb,z_depth)).sum()

        return latent_loss
class dw_con(nn.Module):
    def __init__(self, nin, nout):
        super(dw_con, self).__init__()
        self.depth = nn.Conv2d(nin, nin, kernel_size=3,padding=1, groups=nin, bias=False)
        self.point = nn.Conv2d(nin, nout, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(nout, nout, 1)
    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

def wavelet(map):
    wave = pywt.dwt2(map.detach().cpu().numpy(), 'haar')
    LL, (HL, LH, HH) = wave
    LL = torch.from_numpy(LL).to(device)
    HL = torch.from_numpy(HL).to(device)
    LH = torch.from_numpy(LH).to(device)
    HH = torch.from_numpy(HH).to(device)
    # LL = torch.from_numpy(LL)
    # HL = torch.from_numpy(HL)
    # LH = torch.from_numpy(LH)
    # HH = torch.from_numpy(HH)
    HF = torch.cat([HL, LH, HH], dim=1)

    return LL, HF

def norm(x):
    x = (x-torch.mean(x))/(torch.max(x)-torch.min(x))
    return x



def ContrastiveLoss(high_fm, high_fp, feature):
        feature = norm(feature)
        positive = norm(high_fp)
        negative = (1 - norm(high_fp)) * high_fm
        # n_vgg, p_vgg, o_vgg = self.vgg(negative), self.vgg(positive), self.vgg(output)
        loss = nn.L1Loss()
        clloss = loss(feature, positive)/loss(feature, negative)
        # for i in range(len(o_vgg)):
        #     loss += self.weights[i] * self.criterion(o_vgg[i], p_vgg[i].detach())/(self.criterion(o_vgg[i], n_vgg[i])+self.criterion(o_vgg[i], n_vgg[i]))
        # print('contrastive loss',loss)
        return clloss

class net(nn.Module):

    def __init__(self, inchannel, channel=64, midchannel=32, d = 64):
        super(net, self).__init__()
        self.inchannel = inchannel
        self.proj_m = nn.Sequential(nn.Conv2d(in_channels = inchannel, out_channels=midchannel, kernel_size=3, padding=1, bias=False),
                                    nn.ReLU(), nn.Conv2d(in_channels=midchannel, out_channels=midchannel, kernel_size=3, padding=1, bias=False))
        self.proj_p = nn.Sequential(nn.Conv2d(in_channels=inchannel, out_channels=midchannel//2, kernel_size=3, padding=1, bias=False),
                                    nn.ReLU(), nn.Conv2d(in_channels=midchannel//2, out_channels=midchannel, kernel_size=3, padding=1, bias=False))
        self.gen_m = nn.Sequential(nn.ConvTranspose2d(in_channels=3*midchannel, out_channels=midchannel,kernel_size=2, stride=2, padding=0), nn.Conv2d(in_channels=midchannel, out_channels=channel, kernel_size=3, padding=1, bias=False),
                                    nn.ReLU(), nn.Conv2d(in_channels=channel, out_channels=midchannel, kernel_size=1, bias=False))
        self.gen_p = nn.Sequential(nn.ConvTranspose2d(in_channels=3*midchannel, out_channels=midchannel,kernel_size=2, stride=2, padding=0),nn.Conv2d(in_channels=midchannel, out_channels=channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(), nn.Conv2d(in_channels=channel, out_channels=midchannel, kernel_size=1, bias=False))

        self.generator1_m = dw_con(midchannel, midchannel)
        self.generator1_p = ResBlock(midchannel,midchannel)
        self.feature1 = dw_con(2*midchannel, midchannel)
        self.generator2_m = dw_con(midchannel, midchannel)
        self.generator2_p = ResBlock(midchannel, midchannel)
        self.feature2 = dw_con(2 * midchannel, midchannel)
        self.generator3_m = dw_con(midchannel, midchannel)
        self.generator3_p = ResBlock(midchannel, midchannel)
        self.feature3 = dw_con(2 * midchannel, midchannel)
        self.generator4_m =dw_con(midchannel, midchannel)
        self.generator4_p = ResBlock(midchannel, midchannel)
        self.feature4 = dw_con(2 * midchannel, midchannel)
        self.generator5_m = dw_con(midchannel, midchannel)
        self.generator5_p = ResBlock(midchannel, midchannel)
        self.feature5 = dw_con(2 * midchannel, midchannel)

        self.update = nn.Sequential(nn.Conv2d(in_channels=midchannel*2, out_channels=midchannel,
                                                       kernel_size=3, stride=1, padding=1),
                                     dw_con(midchannel, midchannel))
        
        ### FusionNet 
        # self.update = None
        
        ### LAGNet 
        # self.update = LACNET()

        self.fusion1 = SpaFre(midchannel)
        self.fusion2 = SpaFre(midchannel)
        self.fusion3 = SpaFre(midchannel)
        self.fusion4 = SpaFre(midchannel)
        self.fusion5 = SpaFre(midchannel)

        self.conv = nn.Conv2d(midchannel, inchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.gen1 = ResBlock(inchannel, midchannel//2)
        self.gen2 = ResBlock(midchannel//2, inchannel)
        self.res_ms = ResBlock(inchannel, inchannel)
        # self.mi = Mutual_info_reg(midchannel, int(midchannel / 4), 16)

        # self.feature_g = nn.Sequential(
        #     nn.Conv2d(in_channels=midchannel, out_channels=midchannel, kernel_size=2, stride=2,padding=0),nn.ReLU())
        # self.feature_p = nn.Sequential(
        #     nn.Conv2d(in_channels=midchannel, out_channels=midchannel, kernel_size=2, stride=2, padding=0), nn.ReLU())
        # self.rep_p = nn.Sequential(nn.Flatten(), nn.Linear((midchannel*d**2)//4, 256))
        # self.rep_g = nn.Sequential(nn.Flatten(), nn.Linear((midchannel*d**2)//4, 256))
        #
        # self.sta_g = statistic(midchannel, rep_dim=256, d=d//2)
        # self.sta_p = statistic(midchannel, rep_dim=256, d=d//2)
        # self.sta_gp = statistic(midchannel, rep_dim=256, d=d//2)
        self.mi1 = mi(midchannel, d)
        self.mi2 = mi(midchannel, d)
        self.mi3 = mi(midchannel, d)
        self.mi4 = mi(midchannel, d)
        self.mi5 = mi(midchannel, d)

    def forward(self, lms, pan, gt):
        outl = []
        fm = self.proj_m(lms)
        mpan = torch.cat([pan] * (self.inchannel), dim=1)
        fp = self.proj_p(mpan)
        PLow, PHigh = wavelet(fp)

        MLow, MHigh = wavelet(fm)

        fm_h = self.gen_m(MHigh)
        fp_h = self.gen_p(PHigh)

        fm_h = self.generator1_m(fm_h)
        outl.append(fm_h)
        fp_h = self.generator1_p(fp_h)
        outl.append(fp_h)

        fg = self.proj_p(gt)
        GLow, GHigh = wavelet(fg)
        fg_h = self.gen_p(GHigh)
        fg_h = self.generator1_p(fg_h)
        outl.append(fg_h)
        # mi1 = self.mi1(fp_h, fg_h)

        feature1 = self.feature1(torch.cat([fm_h, fp_h], dim=1))
        cl1 = ContrastiveLoss(fm_h, fp_h, feature1)
        fm_h = self.generator2_m(fm_h)
        fp_h = self.generator2_p(fp_h)
        fg_h = self.generator2_p(fg_h)
        # mi2 = self.mi2(fp_h, fg_h)

        feature2 = self.feature2(torch.cat([fm_h, fp_h], dim=1))
        cl2 = ContrastiveLoss(fm_h, fp_h, feature2)
        fm_h = self.generator3_m(fm_h)
        fp_h = self.generator3_p(fp_h)
        fg_h = self.generator3_p(fg_h)
        # mi3 = self.mi3(fp_h, fg_h)
        
        feature3 = self.feature3(torch.cat([fm_h, fp_h], dim=1))
        cl3 = ContrastiveLoss(fm_h, fp_h, feature3)
        fm_h = self.generator4_m(fm_h)
        fp_h = self.generator4_p(fp_h)
        fg_h = self.generator4_p(fg_h)
        # mi4 = self.mi4(fp_h, fg_h)
        
        feature4 = self.feature4(torch.cat([fm_h, fp_h], dim=1))
        cl4 = ContrastiveLoss(fm_h, fp_h, feature4)
        fm_h = self.generator5_m(fm_h)
        fp_h = self.generator5_p(fp_h)
        fg_h = self.generator5_p(fp_h)
        # mi5 = self.mi5(fp_h, fg_h)
        
        feature5 = self.feature5(torch.cat([fm_h, fp_h], dim=1))
        cl5 = ContrastiveLoss(fm_h, fp_h, feature5)
        clloss = 1.0/32*cl1+1.0/16*cl2+1.0/8*cl3+1.0/4*cl4+1.0/2*cl5
        # mi = 1.0/16*mi1+1.0/8*mi2+1.0/4*mi3+1.0/2*mi4+mi5
        f = torch.cat([fm, fp], dim=1)
        f = self.update(f)
        # f = self.update(pan, lms)

        f_fuse = self.fusion1(f, feature1)
        outl.append(f_fuse)
        f_fuse = self.fusion2(f_fuse, feature2)
        f_fuse = self.fusion3(f_fuse, feature3)
        f_fuse = self.fusion4(f_fuse, feature4)
        f_fuse = self.fusion5(f_fuse, feature5)



        f_fuse = self.conv(f_fuse)

        ms = self.res_ms(lms)
        f_fuse = f_fuse + ms
        output = self.gen1(f_fuse)
        output = self.gen2(output)

        
        # fg_h = self.generator3_p(fg_h)
        # fg_h = self.generator4_p(fg_h)
        # fg_h = self.generator5_p(fg_h)
        # mi = self.mi(fg_h, fp_h)
        # loss = nn.L1Loss()

# semantic loss
        
        # return output, clloss, mi
        return outl, PHigh, MHigh, GHigh
        
class mi(nn.Module):
    def __init__(self, midchannel, d):
        super(mi, self).__init__()
        self.feature_g = nn.Sequential(
            nn.Conv2d(in_channels=midchannel, out_channels=midchannel, kernel_size=2, stride=2, padding=0), nn.ReLU())
        self.feature_p = nn.Sequential(
            nn.Conv2d(in_channels=midchannel, out_channels=midchannel, kernel_size=2, stride=2, padding=0), nn.ReLU())
        self.rep_p = nn.Sequential(nn.Flatten(), nn.Linear((midchannel * d ** 2) // 4, 256))
        self.rep_g = nn.Sequential(nn.Flatten(), nn.Linear((midchannel * d ** 2) // 4, 256))

        self.sta_g = statistic(midchannel, rep_dim=256, d=d // 2)
        self.sta_p = statistic(midchannel, rep_dim=256, d=d // 2)
        self.sta_gp = statistic(midchannel, rep_dim=256, d=d // 2)
    def forward(self, fp_h, fg_h):
        feature_ph = self.feature_p(fp_h)
        rep_ph = self.rep_p(feature_ph)
        feature_gh = self.feature_g(fg_h)
        rep_gh = self.rep_g(feature_gh)
        T = self.sta_g(feature_gh, rep_gh) + self.sta_p(feature_ph, rep_ph)
        T_prime = (self.sta_gp(feature_gh, rep_ph) + self.sta_gp(feature_ph, rep_gh)) / 2
        loss = DJSLoss()
        mi = loss(T, T_prime)
        return mi




class statistic(nn.Module):
    def __init__(self, midchannel, rep_dim, d):
        super(statistic, self).__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=(midchannel*d**2+rep_dim), out_features=64)
        # self.dense2 = nn.Linear(64, 32)
        self.dense2 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
    def forward(self, feature, representation):
        feature = self.flatten(feature)
        x = torch.cat([feature, representation], dim=1)
        x = self.dense1(x)
        # x = self.(x)
        x = torch.sigmoid(x)
        x = self.dense2(x)
        # x = self.relu(x)
        # x = self.dense3(x)
        return x




class ResBlock(nn.Module):
    def __init__(self, nin, nout):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(nin, nin, 1), nn.ReLU(), nn.Conv2d(nin, nout, 3, 1, padding=1))
        self.conv = nn.Conv2d(nin, nout, 1)

    def forward(self, x):
        return self.layer(x) + self.conv(x)




if __name__ == '__main__':
    from torchsummary import summary
    n = net(8, 64).to(device)
    lms = 0.5+torch.randn(32, 8, 64, 64).to(device)
    pan = 0.5+torch.randn(32, 1, 64, 64).to(device)
    gt = 0.5 + torch.randn(32, 8, 64, 64).to(device)
    output = n(lms, pan, gt)
    # summary(n, [(8, 64, 64), (1, 64, 64), (8, 64, 64)])