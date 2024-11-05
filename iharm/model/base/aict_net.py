import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from iharm.model.base.ssam_model import SpatialSeparatedAttention
from iharm.model.modeling.conv_autoencoder import ConvEncoder, DeconvDecoderUpsample
from iharm.model.modeling.unet import UNetEncoder, UNetDecoderUpsample
from iharm.model.modeling.vit_base import ViT_encoder, ViT_decoder
import matplotlib.pyplot as plt


# cascade 0830

class AICT(nn.Module):

    def __init__(
            self,
            backbone_type='idih',
            input_normalization={'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
            linear_num=8, clamp=True, color_space='RGB', use_attn=False,
            depth=4, norm_layer=nn.BatchNorm2d, batchnorm_from=0, attend_from=-1,
            image_fusion=False, ch=64, max_channels=512, attention_mid_k=2.0,
            backbone_from=-1, backbone_channels=None, backbone_mode='',
    ):
        super(AICT, self).__init__()

        self.linear_num = linear_num
        self.clamp = clamp
        self.color_space = color_space
        self.use_attn = use_attn

        self.mean = torch.tensor(input_normalization['mean'], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(input_normalization['std'], dtype=torch.float32).view(1, 3, 1, 1)

        self.dims = 3 * self.linear_num

        self.backbone_type = backbone_type
        input_dim = 32
        bottom_dim = 256
        if backbone_type == 'idih':
            self.encoder = ConvEncoder(
                depth, ch,
                norm_layer, batchnorm_from, max_channels,
                backbone_from, backbone_channels, backbone_mode
            )
            self.decoder = DeconvDecoderUpsample(depth, self.encoder.blocks_channels, norm_layer, attend_from,
                                                 image_fusion)
            input_dim = 32
        elif backbone_type == 'ssam':
            print('depth', depth)
            self.encoder = UNetEncoder(
                depth, ch,
                norm_layer, batchnorm_from, max_channels,
                backbone_from, backbone_channels, backbone_mode
            )
            self.decoder = UNetDecoderUpsample(
                depth, self.encoder.block_channels,
                norm_layer,
                attention_layer=partial(SpatialSeparatedAttention, mid_k=attention_mid_k),
                attend_from=attend_from,
                image_fusion=image_fusion
            )
            input_dim = 32
        elif backbone_type == 'ViT':
            input_dim = 12
            self.encoder = ViT_encoder()
            self.decoder = ViT_decoder(input_dim)

        self.bottom_convs = nn.Sequential(
            nn.Conv2d(bottom_dim, bottom_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bottom_dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(bottom_dim // 2, bottom_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bottom_dim // 4),
            nn.ReLU(inplace=True),
        )
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.class_layer = nn.Linear(in_features=bottom_dim // 4, out_features=self.dims)

        self.colors_layer = nn.Conv2d(in_channels=input_dim, out_channels=self.linear_num * 3, kernel_size=1)

        self.params_layer = nn.Conv2d(in_channels=input_dim, out_channels=self.dims, kernel_size=1)

    def init_device(self, input_device):
        self.device = input_device
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)

    def normalize(self, tensor):
        self.init_device(tensor.device)
        return (tensor - self.mean) / self.std

    def denormalize(self, tensor):
        self.init_device(tensor.device)
        return tensor * self.std + self.mean

    def get_coord(self, x):
        B, _, H, W = x.size()

        coordh, coordw = torch.meshgrid([torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)])
        coordh = coordh.unsqueeze(0).unsqueeze(1).repeat(B, 1, 1, 1).to(x.device)
        coordw = coordw.unsqueeze(0).unsqueeze(1).repeat(B, 1, 1, 1).to(x.device)

        return coordw.detach(), coordh.detach()

    def get_new_coord(self, coordw, coordh, coord_dis):
        B = coord_dis.shape[0]
        H, W = coord_dis.shape[-2:]

        # remap
        x_remap = torch.cumsum(coord_dis[:, :, [0], :, :], dim=4)  # [1, 3, 1, H, W]
        x_zero = torch.zeros([B, 3, 1, H, 1], device=coord_dis.device).detach()
        x_remap = (torch.cat([x_zero, x_remap], dim=4) - 0.5) * 2  # [1, 3, 1, H, W + 1]

        y_remap = torch.cumsum(coord_dis[:, :, [1], :, :], dim=3)  # [1, 3, 1, H, W]
        y_zero = torch.zeros([B, 3, 1, 1, W], device=coord_dis.device).detach()
        y_remap = (torch.cat([y_zero, y_remap], dim=3) - 0.5) * 2  # [1, 3, 1, H+1, W]

        coordw_new_list = []
        coordh_new_list = []
        for i in range(coord_dis.shape[1]):
            x_remap_i = x_remap[:, i, :, :, :]  # [1, 1, H, W]
            y_remap_i = y_remap[:, i, :, :, :]  # [1, 1, H, W]
            x_y_i = torch.stack([coordw, coordh], dim=4).squeeze(1)

            coordw_new = F.grid_sample(x_remap_i, x_y_i, 'bilinear', 'border', True)
            coordh_new = F.grid_sample(y_remap_i, x_y_i, 'bilinear', 'border', True)

            coordw_new_list.append(coordw_new)
            coordh_new_list.append(coordh_new)

        return coordw_new_list, coordh_new_list

    def get_new_color(self, grid_list, color_dis):
        B, _, _, H, W = color_dis.shape

        # remap
        color_remap = torch.cumsum(color_dis, dim=2)   # [1, 3, 8, H, W]
        color_zero = torch.zeros([B, 3, 1, H, W], device=color_dis.device)
        color_remap = torch.cat([color_zero, color_remap], dim=2)  # [1, 3, 9, H, W]
        color_remap = (color_remap - 0.5) * 2   # -1, 1

        x_new_list = []
        for i in range(color_remap.shape[1]):  # [1, 3, 9, H, W]
            color_remap_i = color_remap[:, [i], :, :, :]   # [1, 1, 9, H, W]
            grid_i = grid_list[i]

            x_i_new = F.grid_sample(color_remap_i, grid_i, 'bilinear', 'border', True)
            x_new_list.append(x_i_new.squeeze(1))

        return x_new_list

    def mapping(self, x, param, color_dis):
        # grid: x, y, z -> w, h, d  ~[-1 ,1]
        x = (x - 0.5) * 2
        coordw_raw, coordh_raw = self.get_coord(x)

        x_raw_list = list(torch.chunk(x.detach(), 3, dim=1))  # [[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]
        grid_raw_list = [torch.stack([coordw_raw, coordh_raw, x_i], dim=4) for x_i in x_raw_list]  # [[1, 1, 256, 256, 3], [1, 1, 256, 256, 3] [1, 1, 256, 256, 3]]

        # color
        x_list = self.get_new_color(grid_raw_list, color_dis)

        # merge
        grid_list = [torch.stack([coordw_raw, coordh_raw, x_i], dim=4) for x_i in x_list]  # [[1, 1, 256, 256, 3], [1, 1, 256, 256, 3] [1, 1, 256, 256, 3]]

        curve = torch.stack(torch.chunk(param, 3, dim=1), dim=1)  # [1, 1, 24, 256, 256]
        curve_list = list(torch.chunk(curve, 3, dim=1))  # [[1, 1, 8, 256, 256] , [1, 1, 8, 256, 256], [1, 1, 8, 256, 256]]

        out = torch.cat([F.grid_sample(curve_i, grid_i, 'bilinear', 'border', True)
                         for curve_i, grid_i in zip(curve_list, grid_list)], dim=1).squeeze(2)

        if self.clamp:
            out = torch.clamp(out, 0, 1)
        out = self.normalize(out)

        return out

    def forward(self, image, image_fullres=None, mask=None, mask_fullres=None, backbone_features=None):

        # Low resolution branch
        x = torch.cat((image, mask), dim=1)

        intermediates = self.encoder(x, backbone_features)
        latent, attention_map = self.decoder(intermediates, image, mask)  # (N, 32, 256, 256), (N, 1, 256, 256)

        bottom_feat = intermediates[0]  # (N, 256, 32, 32)

        # class
        class_feat = self.bottom_convs(bottom_feat)
        class_feat = self.max_pool(class_feat).view(class_feat.shape[0], -1)
        class_weight = self.class_layer(class_feat).view(class_feat.shape[0], -1, 1, 1)  # (N, dims, 1, 1)

        # color
        color_dis = self.colors_layer(latent)  # [1, 24, H, W]
        B, _, H, W = color_dis.shape
        color_dis = color_dis.reshape(B, 3, self.linear_num, H, W)  # [1, 3, 8, H, W]
        color_dis = torch.softmax(color_dis, dim=2)  # [1, 3, 8, H, W]

        # param
        params = self.params_layer(latent)

        params = params * class_weight

        output_lowres = self.mapping(self.denormalize(image), params, color_dis)  # input [0, 1]
        output_lowres = output_lowres * attention_map + image * (1 - attention_map)

        outputs = dict()
        outputs['color_dis'] = color_dis
        outputs['images'] = output_lowres
        outputs['params'] = params
        outputs['attention'] = attention_map
        outputs['class_weight'] = class_weight

        # Full resolution branch
        if torch.is_tensor(image_fullres):
            fr_imgs = [image_fullres]
            fr_masks = [mask_fullres]
            idx = [[n for n in range(image_fullres.shape[0])]]
        else:
            fr_imgs = [img_fr.unsqueeze(0).to(image.get_device()) for img_fr in image_fullres]
            fr_masks = [mask_fr.unsqueeze(0).to(image.get_device()) for mask_fr in mask_fullres]
            idx = [[n] for n in range(len(image_fullres))]

        out_fr = []
        for id, fr_img, fr_mask in zip(idx, fr_imgs, fr_masks):
            output_fullres = self.mapping(self.denormalize(fr_img), params[id], color_dis[id])  # [0, 1]
            output_fullres = output_fullres * fr_mask + fr_img * (1 - fr_mask)
            out_fr.append(output_fullres.squeeze())

        if len(out_fr) == 1:
            out_fr = out_fr[0]
        outputs['images_fullres'] = out_fr

        return outputs
