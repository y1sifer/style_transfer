from .models import *
from .utils import *

import os

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from tensorboardX import SummaryWriter

import random
import shutil
from glob import glob
from tqdm import tqdm

from .utils import *
from .models import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base = 16





class StyleTransfer:
    def __init__(self, content_path, style_path):
        self.vgg16 = VGG(models.vgg16(pretrained=True).features[0:23]).to(device).eval()
        self.transform_net = TransformNet(base).to(device)
        self.metanet = MetaNet(self.transform_net.get_param_dict()).to(device)
        self.content_img =  read_image(content_path,target_width=256)
        self.style_img = read_image(style_path, target_width=256)



    def load_model(self):
        self.metanet.load_state_dict(torch.load('/Users/2black/2black_workspace/django_test1/mysite/style_transfer/model/metanet.pth', map_location=lambda storage, loc: storage))
        self.transform_net.load_state_dict(torch.load('/Users/2black/2black_workspace/django_test1/mysite/style_transfer/model/transform_net.pth', map_location=lambda storage, loc: storage))

    def train(self):
        style_weight = 50
        content_weight = 1
        tv_weight = 1e-6
        batch_size = 8
        data_transform = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(256 / 480, 1), ratio=(1, 1)),
            transforms.ToTensor(),
            tensor_normalizer
        ])
        content_dataset = torchvision.datasets.ImageFolder('../data/content/', transform=data_transform)
        trainable_params = {}
        trainable_param_shapes = {}
        for model in [self.vgg16, self.transform_net, self.metanet]:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    trainable_params[name] = param
                    trainable_param_shapes[name] = param.shape

        optimizer = optim.Adam(trainable_params.values(), 1e-3)
        content_data_loader = torch.utils.data.DataLoader(content_dataset, batch_size=batch_size, shuffle=True)

        style_image = self.style_img
        style_features = self.vgg16(style_image)
        style_mean_std = mean_std(style_features)

        n_batch = 20
        with tqdm(enumerate(content_data_loader), total=n_batch) as pbar:
            for batch, (content_images, _) in pbar:
                x = content_images.cpu().numpy()
                if (x.min(-1).min(-1) == x.max(-1).max(-1)).any():
                    continue

                optimizer.zero_grad()

                # 使用风格图像生成风格模型
                weights = self.metanet.forward2(mean_std(style_features))
                self.transform_net.set_weights(weights, 0)

                # 使用风格模型预测风格迁移图像
                content_images = content_images.to(device)
                transformed_images = self.transform_net(content_images)

                # 使用 vgg16 计算特征
                content_features = self.vgg16(content_images)
                transformed_features = self.vgg16(transformed_images)
                transformed_mean_std = mean_std(transformed_features)

                # content loss
                content_loss = content_weight * F.mse_loss(transformed_features[2], content_features[2])

                # style loss
                style_loss = style_weight * F.mse_loss(transformed_mean_std,
                                                       style_mean_std.expand_as(transformed_mean_std))

                # total variation loss
                y = transformed_images
                tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                                       torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

                # 求和
                loss = content_loss + style_loss + tv_loss

                loss.backward()
                optimizer.step()

                if batch > n_batch:
                    break


    # 生成图片
    def transformed_image(self):
        features = self.vgg16(self.style_img)
        mean_std_features = mean_std(features)
        weights = self.metanet.forward2(mean_std_features)
        self.transform_net.set_weights(weights)
        transformed_image = self.transform_net(self.content_img)
        return transformed_image

    def save_image(self, image, path):
        image = recover_image(image)
        plt.imsave(path,image)

