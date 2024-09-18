import numpy as np
import torch
import torch.nn as nn
import torchvision

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


# 实现特征编码器
# class fea_encoder(nn.Module):
#     def __init__(self,dim1):
#         super(fea_encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(dim1, 4096),
#             nn.BatchNorm1d(4096),
#             nn.ReLU(inplace=False),
#             nn.Dropout(0.5)
#         )
#     def forward(self,x):
#         x = self.encoder(x)
#         return x


class fea_encoder(nn.Module):
    def __init__(self, dim1):
        super(fea_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim1, dim1),
            nn.BatchNorm1d(dim1),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(dim1, int(dim1 / 2)),
            nn.BatchNorm1d(int(dim1 / 2)),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(int(dim1 / 2), int(dim1 / 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    def forward(self,x):
        x = self.encoder(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, dim1):
        super(feat_classifier, self).__init__()
        self.classifier_layer = nn.Sequential(
            nn.Linear(int(dim1 / 2), class_num)
        )
        self.classifier_layer.apply(init_weights)

    def forward(self, x):
        out = self.classifier_layer(x)
        return out


class hash_encoder(nn.Module):
    def __init__(self,nbits,dim1):
        super(hash_encoder, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(int(dim1 / 2), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, nbits),
            nn.Tanh()
        )
    def forward(self,x):
        x = self.projector(x)
        return x
