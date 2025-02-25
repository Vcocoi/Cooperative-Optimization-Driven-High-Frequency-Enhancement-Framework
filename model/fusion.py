import torch.nn as nn
import torch
class SpaFre(nn.Module):
    def __init__(self, channels):
        super(SpaFre, self).__init__()
        self.spa_att = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.cha_att = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, kernel_size=1, padding=0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels * 2, kernel_size=1, padding=0, bias=True),
                                     nn.Sigmoid())
        self.post = nn.Conv2d(channels * 2, channels, 3, 1, 1)

    def forward(self, spafuse, frefuse):
        spa_map = self.spa_att(spafuse-frefuse)
        spa_res = frefuse*spa_map+spafuse
        cat_f = torch.cat([spa_res,frefuse],1)
        cha_res =  self.post(self.cha_att(self.contrast(cat_f) + self.avgpool(cat_f))*cat_f)
        out = cha_res+spafuse
        return out
        

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))
def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)