import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class HSI_Attention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, window_size, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B,C,H,W = x.shape
        x = x.permute(0,2,3,1) # B H W C
        x_windows = window_partition(x, self.window_size)  # B H W C ---> num_windows*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  #num_windows*B, window_size*window_size, C
        qk = self.qkv(x_windows).reshape(-1, self.window_size ** 2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #num_windows*B, window_size*window_size, C -->{qkv:Linear}--> num_windows*B, window_size*window_size, 2C
        # ---->{reshape}-->window_size*window_size,2,num_heads, C//num_heads
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        #x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        #x = self.proj(x)
        #x = self.proj_drop(x)
        return attn
if __name__ =="__main__":
    input = torch.randn(1,64,64*64) #B C (HW)
    net = HSI_Attention(dim=64,heads=8,dim_head=8)
    out = net(input)
    print(out.shape)