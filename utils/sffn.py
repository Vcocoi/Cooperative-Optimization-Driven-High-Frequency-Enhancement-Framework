import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange


def make_model(args, parent=False):
    return SFFN()


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window=3, focal_level=3, focal_factor=2, bias=True):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor

        self.f = nn.Conv2d(dim, 2*dim + (self.focal_level+1), 1, 1, 0, bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.focal_layers = nn.ModuleList()
                
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, 
                    groups=dim, padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                )              
            self.kernel_sizes.append(kernel_size)          

    def forward(self, x):
        C = x.shape[1]

        # pre linear projection
        x = self.f(x)
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level+1), 1)
        
        # context aggreation
        ctx_all = 0 
        for l in range(self.focal_level):         
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx*self.gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        # print(self.gates[:,self.focal_level:].size(), ctx_global.size())
        ctx_all = ctx_all + ctx_global*self.gates[:,self.focal_level:]

        # focal modulation
        self.modulator = self.h(ctx_all)
        x_out = q*self.modulator

        # post linear porjection
        x_out = self.proj(x_out)
        return x_out


class SpaBlock(nn.Module):
    def __init__(self, in_size, out_size, chnum=4):
        super(SpaBlock, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        if self.in_size != self.out_size:
            self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.layer_1 = nn.Sequential(*[
            LayerNorm2d(out_size),
            FocalModulation(out_size)
        ])
        self.layer_2 = nn.Sequential(*[
            nn.Conv2d(out_size,out_size,3,1,1),
            nn.GELU(),
            nn.Conv2d(out_size, out_size,3,1,1),
            nn.GELU()
        ])
        
    def forward(self, x):
        if self.in_size != self.out_size:
            x = self.identity(x)
        # print(x.shape)
        x = self.layer_1(x) + x
        x = self.layer_2(x) + x
        return x


class MFRFC(nn.Module):
    def __init__(self, in_channels, order=0.25):
        super(MFRFC, self).__init__()
        self.order = order
        C0 = int(in_channels / 3)
        C1 = int(in_channels) - 2 * C0
        self.conv_0 = nn.Conv2d(C0, C0, kernel_size=3, padding=1)
        self.conv_05 = nn.Conv2d(2 * C1, 2 * C1, kernel_size=1, padding=0)
        self.conv_1 = nn.Conv2d(2 * C0, 2 * C0, kernel_size=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def dfrtmtrx(self, N, a):
        # Approximation order
        app_ord = 2
        Evec = self.dis_s(N, app_ord)
        Evec = Evec.to(dtype=torch.complex64)
        even = 1 - (N % 2)
        l = torch.tensor(list(range(0, N - 1)) + [N - 1 + even])
        f = torch.diag(torch.exp(-1j * math.pi / 2 * a * l))
        F = N ** (1 / 2) * torch.einsum("ij,jk,ni->nk", f, Evec.T, Evec)
        return F

    def dis_s(self, N, app_ord):
        app_ord = int(app_ord / 2)
        s = torch.cat((torch.tensor([0, 1]), torch.zeros(N - 1 - 2 * app_ord), torch.tensor([1])))
        S = self.cconvm(N, s) + torch.diag((torch.fft.fft(s)).real)

        p = N
        r = math.floor(N / 2)
        P = torch.zeros((p, p))
        P[0, 0] = 1
        even = 1 - (p % 2)

        for i in range(1, r - even + 1):
            P[i, i] = 1 / (2 ** (1 / 2))
            P[i, p - i] = 1 / (2 ** (1 / 2))

        if even:
            P[r, r] = 1

        for i in range(r + 1, p):
            P[i, i] = -1 / (2 ** (1 / 2))
            P[i, p - i] = 1 / (2 ** (1 / 2))

        CS = torch.einsum("ij,jk,ni->nk", S, P.T, P)
        C2 = CS[0:math.floor(N / 2 + 1), 0:math.floor(N / 2 + 1)]
        S2 = CS[math.floor(N / 2 + 1):N, math.floor(N / 2 + 1):N]
        ec, vc = torch.linalg.eig(C2)
        es, vs = torch.linalg.eig(S2)
        ec = ec.real
        vc = vc.real
        es = es.real
        vs = vs.real
        qvc = torch.vstack((vc, torch.zeros([math.ceil(N / 2 - 1), math.floor(N / 2 + 1)])))
        SC2 = P @ qvc  # Even Eigenvector of S
        qvs = torch.vstack((torch.zeros([math.floor(N / 2 + 1), math.ceil(N / 2 - 1)]), vs))
        SS2 = P @ qvs  # Odd Eigenvector of S
        idx = torch.argsort(-ec)
        SC2 = SC2[:, idx]
        idx = torch.argsort(-es)
        SS2 = SS2[:, idx]

        if N % 2 == 0:
            S2C2 = torch.zeros([N, N + 1])
            SS2 = torch.hstack([SS2, torch.zeros((SS2.shape[0], 1))])
            S2C2[:, range(0, N + 1, 2)] = SC2
            S2C2[:, range(1, N, 2)] = SS2
            S2C2 = S2C2[:, torch.arange(S2C2.size(1)) != N - 1]
        else:
            S2C2 = torch.zeros([N, N])
            S2C2[:, range(0, N + 1, 2)] = SC2
            S2C2[:, range(1, N, 2)] = SS2

        return S2C2

    def cconvm(self, N, s):
        M = torch.zeros((N, N))
        dum = s
        for i in range(N):
            M[:, i] = dum
            dum = torch.roll(dum, 1)
        return M

    def FRFT2D(self, matrix):
        N, C, H, W = matrix.shape
        h_test = self.dfrtmtrx(H, self.order).cuda() # .to('cuda:1')
        w_test = self.dfrtmtrx(W, self.order).cuda() # .to('cuda:1')
        h_test = torch.repeat_interleave(h_test.unsqueeze(dim=0), repeats=C, dim=0)
        h_test = torch.repeat_interleave(h_test.unsqueeze(dim=0), repeats=N, dim=0)
        w_test = torch.repeat_interleave(w_test.unsqueeze(dim=0), repeats=C, dim=0)
        w_test = torch.repeat_interleave(w_test.unsqueeze(dim=0), repeats=N, dim=0)

        out = []
        matrix = torch.fft.fftshift(matrix, dim=(2, 3)).to(dtype=torch.complex64)

        out = torch.matmul(h_test, matrix)
        out = torch.matmul(out, w_test)

        out = torch.fft.fftshift(out, dim=(2, 3))
        return out

    def IFRFT2D(self, matrix):
        N, C, H, W = matrix.shape
        h_test = self.dfrtmtrx(H, -self.order).cuda() # .to('cuda:1')
        w_test = self.dfrtmtrx(W, -self.order).cuda() # .to('cuda:1')
        h_test = torch.repeat_interleave(h_test.unsqueeze(dim=0), repeats=C, dim=0)
        h_test = torch.repeat_interleave(h_test.unsqueeze(dim=0), repeats=N, dim=0)
        w_test = torch.repeat_interleave(w_test.unsqueeze(dim=0), repeats=C, dim=0)
        w_test = torch.repeat_interleave(w_test.unsqueeze(dim=0), repeats=N, dim=0)

        out = []
        matrix = torch.fft.fftshift(matrix, dim=(2, 3)).to(dtype=torch.complex64)

        out = torch.matmul(h_test, matrix)
        out = torch.matmul(out, w_test)

        out = torch.fft.fftshift(out, dim=(2, 3))
        return out

    def forward(self, x):
        N, C, H, W = x.shape

        C0 = int(C / 3)
        x_0 = x[:, 0:C0, :, :]
        x_05 = x[:, C0:C - C0, :, :]
        x_1 = x[:, C - C0:C, :, :]

        # order = 0
        x_0 = self.conv_0(x_0)

        # order = 0.5
        Fre = self.FRFT2D(x_05)
        Real = Fre.real
        Imag = Fre.imag
        Mix = torch.concat((Real, Imag), dim=1)
        Mix = self.conv_05(Mix)
        Real1, Imag1 = torch.chunk(Mix, 2, 1)
        Fre_out = torch.complex(Real1, Imag1)
        IFRFT = self.IFRFT2D(Fre_out)
        IFRFT = torch.abs(IFRFT) / (H * W)

        # order = 1
        fre = torch.fft.rfft2(x_1, norm='backward')
        real = fre.real
        imag = fre.imag
        mix = torch.concat((real, imag), dim=1)
        mix = self.conv_1(mix)
        real1, imag1 = torch.chunk(mix, 2, 1)
        fre_out = torch.complex(real1, imag1)
        x_1 = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')

        output = torch.cat([x_0, IFRFT, x_1], dim=1)
        output = self.conv2(output)

        return output


class UncertaintyEstimation(nn.Module):
    def __init__(self, channels, T, q, cnum=3):
        super(UncertaintyEstimation, self).__init__()
        self.T = T
        self.q = q
        self.conv = nn.Sequential(nn.Conv2d(channels, channels * 2, 3, 1, 1, bias=True))
        self.out = nn.Sequential(nn.Conv2d(channels * 2, cnum, 1, 1, 0), nn.Tanh())
        self.aue = nn.Sequential(nn.Conv2d(channels * 2, cnum, 3, 1, 1), nn.Sigmoid())

    def random_mask(self, x, q):
        mask = np.random.binomial(n=1, p=1 - q, size=(self.T, x.shape[1], x.shape[2]))
        mask = torch.tensor(mask).to(x.device)
        mask = rearrange(mask, "T B C -> T B C 1 1")
        return x * mask

    def epistemic_uncetainty(self, x):
        mean = 0
        # xs = []
        # for i in range(self.T):
        #     x_cur = self.out(self.random_mask(x, self.q))
        #     x_cur = rearrange(x_cur, "B C H W -> 1 B C H W")
        #     xs.append(x_cur)
        # xs = torch.cat(xs, dim=0)
        xs = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        xs = self.random_mask(xs, self.q)
        t, b, c, h, w = xs.shape
        xs = xs.reshape(t*b, c, h, w)
        xs = self.out(xs).reshape(t, b, -1, h, w)

        EU, mean = torch.var_mean(input=xs, dim=0, unbiased=True)

        return EU, mean

    # def aleatoric_uncetainty(self, x):
    #     return self.aue(x)

    def forward(self, x):
        x = self.conv(x)
        EU, mean = self.epistemic_uncetainty(x)
        # AU = self.aleatoric_uncetainty(x)
        return EU, mean # AU,


class MergeBlock(nn.Module):
    # Spa分支融合
    # Fre分支融合
    def __init__(self, in_ch, out_ch):
        super(MergeBlock, self).__init__()
        self.mean_conv = nn.Conv2d(3, in_ch, 3, 1, 1)
        self.merge = nn.Sequential(nn.Conv2d(2*in_ch, out_ch, 1, 1), nn.GELU(),
                                   nn.Conv2d(out_ch, out_ch, 3, 1, 1))

    def forward(self, x, mean):
        mean = self.mean_conv(mean)
        out = torch.cat([mean, x], dim=1)
        out = self.merge(out)

        return out


class BasicBlock(nn.Module):
    # encoder 阶段：the same number of channel in input and output
    # decoder 阶段：the various channel in input and output 
    def __init__(self, in_size, out_size, down = False, chnum=4, order = 0.5):
        super(BasicBlock, self).__init__()
        self.downsample = down
        self.spa_pro = SpaBlock(in_size=in_size, out_size=in_size)
        self.spa_ue = UncertaintyEstimation(in_size, T=5, q=0.2)
        self.spa_merge = MergeBlock(in_size, in_size)

        self.fre_preconv = nn.Conv2d(in_size, in_size, 3, 1, 1)
        self.fre_gate = nn.GELU()
        self.fre_pro = MFRFC(in_channels=in_size, order=order)
        self.fre_ue = UncertaintyEstimation(in_size, T=5, q=0.2)
        self.fre_merge = MergeBlock(in_size, in_size)
        self.out_conv = nn.Sequential(nn.Conv2d(in_size, in_size, 1, 1), nn.GELU(),
                                      nn.Conv2d(in_size, in_size, 3, 1, 1))
        if self.downsample:
            # self.down = DownSample(out_size, 2)
            self.down = nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=True)


    def forward(self, x):
        # print(x.shape)
        x_spa = self.spa_pro(x)
        spa_eu, spa_mean = self.spa_ue(x_spa) # spa_au,
        x_smge = self.spa_merge(x_spa, spa_mean)
        # x_smge = x_spa

        x_res = self.fre_preconv(x_smge)
        x_fre1 = self.fre_gate(x_res)
        x_fre2 = self.fre_pro(x_fre1)
        x_fre = x_fre2 + x_res
        fre_eu, fre_mean = self.fre_ue(x_fre) # fre_au,
        x_fmge = self.fre_merge(x_fre, fre_mean)
        # x_fmge = x_fre
        x_fuse = x_spa + x_fmge
        # print(x.shape, map.shape)
        out = self.out_conv(x_fuse)

        if self.downsample:
            out_down = self.down(out)
            return out, out_down, spa_mean, fre_mean
        else:
            return out, spa_mean, fre_mean


# for cross-scale interaction (CSI) module 
# ======================================BEGIN======================================


class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels*chan_factor), 1, stride=1, padding=0, bias=bias)
        )
        
    def forward(self, x):
        return self.bot(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=True):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels//chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        )

    def forward(self, x):
        return self.bot(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

# ======================================END======================================


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, order = 0.5):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        # self.up = nn.Sequential(nn.Conv2d(in_size, out_size*4, kernel_size=3, stride=1, padding=1, bias=False),
        #                           nn.PixelShuffle(2))
        self.fuse = nn.Conv2d(2*out_size, out_size, 3, 1, 1)
        self.conv_block = BasicBlock(out_size, out_size, order=order)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.fuse(out)
        out = self.conv_block(out)
        return out


class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(n_feat, 3, 3, 1, 1)
        self.conv3 = nn.Conv2d(3, n_feat,3, 1, 1)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1 + x
        return x1, img


class SFFN(nn.Module):
    def __init__(self, in_chn=3, wf=20, depth=4, channel=4, order=0.5):
        super(SFFN, self).__init__()
        self.depth = depth
        self.encoder = nn.ModuleList()
        self.downlist = nn.ModuleList()
        self.chlist = []
        self.conv0 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.sam = SAM(wf)

        prev_channels = wf
        for i in range(depth):
            down = True if (i + 1) < depth else False
            self.encoder.append(BasicBlock(prev_channels, (2**(i+1)) * wf, down=down, order=order))
            prev_channels = (2**(i+1)) * wf

        prev_channels = (2**(depth-1)) * wf

        self.decoder = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.decoder.append(UNetUpBlock(prev_channels, (2**i)*wf))
            prev_channels = (2**i)*wf

    def forward(self, x):
        image = x
        x = self.conv0(x)
        encs = []
        spa_outs = []
        spa_aouts = []
        spa_pouts = []
        fre_outs = []
        fre_aouts = []
        fre_pouts = []
        # decs = []
        for i, enc in enumerate(self.encoder):
            if (i+1) < self.depth:
                x_up, x, x_spa, x_fre = enc(x)
                encs.append(x_up)
                spa_outs.append(x_spa)
                fre_outs.append(x_fre)

                # x_sfft = torch.fft.rfft2(x_spa)
                # x_sa = torch.abs(x_sfft)
                # x_sp = torch.angle(x_sfft)
                # spa_aouts.append(x_sa)
                # spa_pouts.append(x_sp)
                #
                # x_ffft = torch.fft.rfft2(x_fre)
                # x_fa = torch.abs(x_ffft)
                # x_fp = torch.angle(x_ffft)
                # fre_aouts.append(x_fa)
                # fre_pouts.append(x_fp)
                # print(x_up.shape, x.shape)
            else:
                x, x_spa, x_fre = enc(x)
                spa_outs.append(x_spa)
                fre_outs.append(x_fre)

                # x_sfft = torch.fft.rfft2(x_spa)
                # x_sa = torch.abs(x_sfft)
                # x_sp = torch.angle(x_sfft)
                # spa_aouts.append(x_sa)
                # spa_pouts.append(x_sp)
                #
                # x_ffft = torch.fft.rfft2(x_fre)
                # x_fa = torch.abs(x_ffft)
                # x_fp = torch.angle(x_ffft)
                # fre_aouts.append(x_fa)
                # fre_pouts.append(x_fp)
                # print(x.shape)

        for i, dec in enumerate(self.decoder):
            x, x_spa, x_fre = dec(x, encs[-i-1])

            spa_outs.append(x_spa)
            fre_outs.append(x_fre)
            #
            # x_sfft = torch.fft.rfft2(x_spa)
            # x_sa = torch.abs(x_sfft)
            # x_sp = torch.angle(x_sfft)
            # spa_aouts.append(x_sa)
            # spa_pouts.append(x_sp)
            #
            # x_ffft = torch.fft.rfft2(x_fre)
            # x_fa = torch.abs(x_ffft)
            # x_fp = torch.angle(x_ffft)
            # fre_aouts.append(x_fa)
            # fre_pouts.append(x_fp)
            # print(x.shape)
            # decs.append(x)

        _, out = self.sam(x, image)

        return out, spa_outs, fre_outs  # , spa_aouts, spa_pouts, fre_aouts, fre_pouts

# class Dehaze(nn.Module):
#     def __init__(self, base_channel=20):
#         super(Dehaze, self).__init__()
#
#         self.first = FirstNet(wf=base_channel)
#
#     def forward(self, x):
#         image = x
#         out_1, encs_first, decs_first = self.first(x)
#
#         out_1_feature, out_1 = self.sam(out_1, image)
#
#         # frequency features exchange
#         # ============================================================================================
#         out_1_fft = torch.fft.rfft2(out_1, norm='backward')
#         out_1_amp = torch.abs(out_1_fft)
#
#         out_1_phase = torch.angle(out_1_fft)
#
#         image_fft = torch.fft.rfft2(image, norm='backward')
#         image_phase = torch.angle(image_fft)
#         image_inverse = torch.fft.irfft2(out_1_amp*torch.exp(1j*image_phase), norm='backward')
#         # ============================================================================================
#
#         out_2 = self.second(out_1_feature, image_inverse, encs_first, decs_first)
#
#         return [out_1, out_1_amp, out_1_phase, out_2, image_phase]

if __name__ == '__main__':
    from torchsummary import summary
    model = SFFN().to('cuda:0')
    # x = torch.randn(1,3,256,256).to('cuda:0')
    # y, spa_outs, fre_outs = model(x)  # , spa_aouts, spa_pouts, fre_aouts, fre_pouts
    # print(y.shape, len(spa_outs), len(fre_outs))  # , len(spa_aouts), len(spa_pouts), len(fre_aouts), len(fre_pouts)
    # for i in range(len(spa_outs)):
    #     print(spa_outs[i].shape, fre_outs[i].shape)
    summary(model, [(3, 256, 256)], device='cuda')
    # from thop import profile
    # flops, params = profile(model, inputs=(x,))
    # print('Params and FLOPs are {}M/{}G'.format(params/1e6, flops/1e9))
    # 3.350289M/18.17919488G
