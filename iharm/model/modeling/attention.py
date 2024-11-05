import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LocalAttention(nn.Module):
    def __init__(self, dims_in, num_heads=8, bias=True):
        super(LocalAttention, self).__init__()
        self.num_heads = num_heads
        self.q_conv = nn.Conv2d(in_channels=dims_in, out_channels=dims_in, kernel_size=1, bias=bias)
        self.k_conv = nn.Conv2d(in_channels=dims_in, out_channels=dims_in, kernel_size=1, bias=bias)
        self.v_conv = nn.Conv2d(in_channels=dims_in, out_channels=dims_in, kernel_size=1, bias=bias)

        self.fusion_conv = nn.Conv2d(in_channels=2 * dims_in, out_channels=dims_in, kernel_size=1, bias=bias)

    def forward(self, x, mask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='bilinear')
        B, C, H, W = x.shape

        num = H * W

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = q.contiguous().view(B * self.num_heads, -1, num).permute(0, 2, 1)  # (Bh, N, E)
        k = k.contiguous().view(B * self.num_heads, -1, num).permute(0, 2, 1)  # (Bh, N, E)
        v = v.contiguous().view(B * self.num_heads, -1, num).permute(0, 2, 1)  # (Bh, N, E)

        E = q.shape[-1]
        q = q / math.sqrt(E)
        # (Bh, N, E) x (Bh, E, N) -> (Bh, N, N)
        attn = torch.bmm(q, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        fuse_feat = torch.bmm(attn, v)  # (Bh, N, E)
        fuse_feat = fuse_feat.permute(0, 2, 1).contiguous().view(B, C, H, W)

        hybrid_feat = torch.cat((x, fuse_feat), 1)
        hybrid_feat = self.fusion_conv(hybrid_feat)

        return hybrid_feat * mask + x * (1 - mask)