import torch
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from torch import nn
import math

class ViT_Harmonizer(nn.Module):
    def __init__(self, output_nc, ksize=4, tr_r_enc_head=2, tr_r_enc_layers=9, input_nc=3, dim_forward=2, tr_act='gelu'):
        super(ViT_Harmonizer, self).__init__()
        dim = 256
        self.patch_to_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = ksize, p2 = ksize),
                nn.Linear(ksize*ksize*(input_nc+1), dim)
            )
        self.transformer_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, nhead=tr_r_enc_head, dim_feedforward=dim*dim_forward, activation=tr_act), num_layers=tr_r_enc_layers)
        self.dec = nn.ConvTranspose2d(dim, output_nc, kernel_size=ksize, stride=ksize, padding=0)

    def forward(self, inputs, backbone_features=None):
        patch_embedding = self.patch_to_embedding(inputs)
        content = self.transformer_enc(patch_embedding.permute(1, 0, 2))
        bs, L, C = patch_embedding.size()
        content = content.permute(1, 2, 0).view(bs, C, int(math.sqrt(L)), int(math.sqrt(L)))
        harmonized = self.dec(content)
        return harmonized


class ViT_encoder(nn.Module):
    def __init__(self, ksize=4, tr_r_enc_head=2, tr_r_enc_layers=9, input_nc=3, dim_forward=2,
                 tr_act='gelu'):
        super(ViT_encoder, self).__init__()
        dim = 256
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=ksize, p2=ksize),
            nn.Linear(ksize * ksize * (input_nc + 1), dim)
        )
        self.transformer_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, nhead=tr_r_enc_head, dim_feedforward=dim * dim_forward, activation=tr_act),
            num_layers=tr_r_enc_layers)

    def forward(self, inputs, backbone_features=None):
        patch_embedding = self.patch_to_embedding(inputs)  # b (h w) (p1 p2 c)
        content = self.transformer_enc(patch_embedding.permute(1, 0, 2))
        bs, L, C = patch_embedding.size()
        content = content.permute(1, 2, 0).view(bs, C, int(math.sqrt(L)), int(math.sqrt(L)))
        return [content]


class ViT_encoder_token(nn.Module):
    def __init__(self, ksize=4, tr_r_enc_head=2, tr_r_enc_layers=9, input_nc=3, dim_forward=2,
                 tr_act='gelu'):
        super(ViT_encoder_token, self).__init__()
        dim = 256
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=ksize, p2=ksize),
            nn.Linear(ksize * ksize * (input_nc + 1), dim)
        )
        self.transformer_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, nhead=tr_r_enc_head, dim_feedforward=dim * dim_forward, activation=tr_act),
            num_layers=tr_r_enc_layers)
        self.token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, inputs, backbone_features=None):
        patch_embedding = self.patch_to_embedding(inputs)  # b (h w) (p1 p2 c)
        b, n, d = patch_embedding.size()
        tokens = repeat(self.token, '() n d -> b n d', b=b)
        x = torch.cat((tokens, patch_embedding), dim=1)  # b, n+1, d
        x = x.permute(1, 0, 2)  # n+1, b, d

        x = self.transformer_enc(x)  # n+1, b, d
        content = x[1:]  # n, b, d
        out_tokens = x[1]  # b, d
        content = content.permute(1, 2, 0).view(b, d, int(math.sqrt(n)), int(math.sqrt(n)))
        return [content], out_tokens


class ViT_decoder(nn.Module):
    def __init__(self, output_nc, ksize=4):
        super(ViT_decoder, self).__init__()
        dim = 256
        self.dec = nn.ConvTranspose2d(dim, output_nc, kernel_size=ksize, stride=ksize, padding=0)

    def forward(self, content_list, input_image, mask):
        content = content_list[0]
        output = self.dec(content)
        return output, mask

