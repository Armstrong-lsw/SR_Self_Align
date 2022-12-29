import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.models.archs.arch_util import default_init_weights, make_layer, ResidualBlockNoBN

from basicsr.models.archs.SEModule import SELayer

from basicsr.models.archs.AlignmentPCD import PCD_Alignment


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in RRDB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1,
                               1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1,
                               1)
        self.conv4_att = SELayer(num_grow_ch, 8)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.conv5_att = SELayer(num_feat, 8)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4_att, self.conv4, self.conv5_att, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x4 = self.conv4_att(x4)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = self.conv5_att(x5)
        # Emperically, we use 0.2 to scale the residual for better performance
        # return x5 * 0.2 + self.conv_res(x)
        return x5 * 0.2 + x


class RRDB2(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in RRDB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB2, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        # return out * 0.2 + self.conv_res(x)
        return out * 0.2 + x


class RRDBDCNNetSESE(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in RRDB.

    RRDB: Enhanced Super-Resolution Generative Adversarial Networks.
    Currently, it supports x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=23,
                 num_grow_ch=32,
                 num_extract_block=5,
                 deformable_groups=8):
        super(RRDBDCNNetSESE, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(
            ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.body = make_layer(
            RRDB2, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                                       SELayer(num_feat,8))
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.pcd_align = PCD_Alignment(num_feat=num_feat, deformable_groups=deformable_groups)
    def forward(self, x):
        feat = self.conv_first(x)

        #dcn
        feat_l1 = self.feature_extraction(feat)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        nbr_feat_l = [  # neighboring feature list
            feat_l1[:, :, :, :].clone(), feat_l2[:, :, :, :].clone(),
            feat_l3[:, :, :, :].clone()
        ]

        aligned_feat=self.pcd_align(nbr_feat_l)

        body_feat = self.conv_body(self.body(aligned_feat))
        feat = feat + body_feat

        # upsample
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
