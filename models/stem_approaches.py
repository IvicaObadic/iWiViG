import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
import matplotlib.pyplot as plt

import captum
from captum.attr import visualization as viz
import numpy as np

from .vig_pytorch.gcn_lib.torch_nn import act_layer
import bagnets.pytorchnet

def get_receptive_field(image, stem_model, node_i_idx=0, node_j_idx=0):
    image_to_attr = copy.deepcopy(image)
    stem_model.zero_grad()
    stem_model.eval()
    #print(stem_model)
    saliency = captum.attr.Saliency(stem_model)
    attribution = saliency.attribute(image_to_attr,
                                     target=(0, node_i_idx, node_j_idx)).squeeze().cpu().detach().numpy()

    attribution_per_pixel = np.sum(np.abs(attribution), axis=0)
    num_pixels_in_rec_field = np.count_nonzero(attribution_per_pixel)

    # print("Node: {}_{}. Receptive field: {}".format(node_i_idx, node_j_idx, num_pixels_in_rec_field))
    # attr_viz = np.transpose(attribution, (1, 2, 0))
    # viz.visualize_image_attr(attr_viz, None, method="heat_map", sign="absolute_value",
    #                          show_colorbar=True, title=" {} Node {},{}".format(stem_model.print(), node_i_idx, node_j_idx))
    non_zero_attributions = np.nonzero(attribution_per_pixel)
    stem_model.zero_grad()
    return list(zip(non_zero_attributions[0].tolist(), non_zero_attributions[1].tolist()))


class IsotropicViGStem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x

    def print(self):
        return "Isotropic_ViG_stem"

class PyramidViGStem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim))

    def forward(self, x):
        x = self.convs(x)
        return x

    def print(self):
        return "Pyramid_ViG_stem"


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768, dropout=0.1):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        B, C, H, W = x.shape
        #x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = F.gelu(self.batch_norm(x))
        x = self.dropout(x)
        return x


class BagNetStem(nn.Module):
    def __init__(self, bagnet_ver="17", embedding_dim=192, use_gelu=True):
        super().__init__()
        if bagnet_ver == "17":
            self.patch_encoder = bagnets.pytorchnet.bagnet17(num_classes=embedding_dim, avg_pool=False)
        else:
            self.patch_encoder = bagnets.pytorchnet.bagnet33(num_classes=embedding_dim, avg_pool=False)
        if use_gelu:
            self.patch_encoder.relu = nn.GELU()
            self.patch_encoder.layer1.relu = nn.GELU()
            self.patch_encoder.layer2.relu = nn.GELU()
            self.patch_encoder.layer3.relu = nn.GELU()
            self.patch_encoder.layer4.relu = nn.GELU()

    def forward(self, x) -> torch.Tensor:
        patch_embeddings = self.patch_encoder(x).permute(0, 3, 1, 2)
        return patch_embeddings

    def print(self):
        return "bagnet_stem"


class BagnetTiny(nn.Module):
    def __init__(self, in_channels = 3, bagnet_ver="17", embedding_dim=192, act_fun="gelu"):
        super().__init__()
        self.act_func = act_layer(act_fun)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embedding_dim // 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim // 8, momentum=0.001),
            self.act_func,
            nn.Conv2d(embedding_dim // 8, embedding_dim // 4, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(embedding_dim // 4, momentum=0.1),
            self.act_func,
            nn.Conv2d(embedding_dim // 4, embedding_dim // 2, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(embedding_dim // 2, momentum=0.1),
            self.act_func,
            nn.Conv2d(embedding_dim // 2, embedding_dim, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(embedding_dim, momentum=0.1),
            self.act_func,
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim, momentum=0.1),
            self.act_func,
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(embedding_dim, momentum=0.1),
            self.act_func)

    def forward(self, x) -> torch.Tensor:
        x = self.patch_embed(x)

    def print(self):
        return "bagnetTiny_stem"


class FishnetDownsample(nn.Module):
    def __init__(self, in_channels = 3, embedding_dim=192, act_fun="gelu"):
        super().__init__()
        self.act_func = act_layer(act_fun)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embedding_dim // 8, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(embedding_dim // 8),
            self.act_func,
            nn.Conv2d(embedding_dim // 8, embedding_dim // 4, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(embedding_dim // 4),
            self.act_func,
            nn.Conv2d(embedding_dim // 4, embedding_dim // 2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(embedding_dim // 2),
            self.act_func,
            nn.Conv2d(embedding_dim // 2, embedding_dim, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(embedding_dim),
            self.act_func)

    def forward(self, x) -> torch.Tensor:
        return self.patch_embed(x)

    def print(self):
        return "fishnetDownsample_stem"

def resolve_stem_module(stem_approach, embedding_dim):
    if stem_approach == "patch_linear":
        return PatchEmbed(embed_dim=embedding_dim), 256
    elif stem_approach == "vig_stem":
        return IsotropicViGStem(out_dim=embedding_dim, act="gelu"), 256
    elif stem_approach == "pyramid_vig_stem":
        return PyramidViGStem(out_dim=embedding_dim, act="gelu"), 4096
    elif stem_approach == "bagnet_stem":
        return BagNetStem(embedding_dim=embedding_dim), 900
    elif stem_approach == "bagnetTiny":
        return BagnetTiny(embedding_dim=embedding_dim), 64
    else:
        return FishnetDownsample(embedding_dim=embedding_dim), 256


if __name__ == '__main__':
    stem_model = BagnetTiny(embedding_dim=192)
    dummy_image = torch.randn((1, 3, 256, 256))
    output = stem_model(dummy_image)
    print(output.shape)
    rec_field = get_receptive_field(dummy_image, stem_model, 0 ,0)
    #print(get_receptive_field(dummy_image, stem_model, 6 ,6))
    #stem_model = IsotropicViGStem(out_dim=192)
