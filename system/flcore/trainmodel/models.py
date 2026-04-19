import math
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.Swin_transformer import *
from flcore.trainmodel.SVD_Swin_transformer import *
from flcore.trainmodel.SVD_resnet import *
from flcore.trainmodel.VIT import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *
import copy
import numpy as np
import string
import math
import torch.nn.utils as utils
# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, args, cid=0, feature_dim=None, is_global=False):
        super().__init__()
        if feature_dim is None:
            feature_dim = args.feature_dim

        if is_global:
            self.base = eval(args.global_model)
        else:
            self.base = eval(args.models[cid % len(args.models)])

        head = None # you may need more code for pre-existing heterogeneous heads
        if hasattr(self.base, 'heads'):
            head = self.base.heads
            self.base.heads = nn.AdaptiveAvgPool1d(feature_dim)
        elif hasattr(self.base, 'head'):
            head = self.base.head
            self.base.head = nn.AdaptiveAvgPool1d(feature_dim)
        elif hasattr(self.base, 'fc'):
            head = self.base.fc
            self.base.fc = nn.AdaptiveAvgPool1d(feature_dim)
        elif hasattr(self.base, 'classifier'):
            head = self.base.classifier
            self.base.classifier = nn.AdaptiveAvgPool1d(feature_dim)
        else:
            raise('The base model does not have a classification head.')

        if hasattr(args, 'heads'):
            self.head = eval(args.heads[cid % len(args.heads)])
        elif 'vit' in args.models[cid % len(args.models)]:
            self.head = nn.Sequential(
                nn.Linear(feature_dim, 768), 
                nn.Tanh(),
                nn.Linear(768, args.num_classes)
            )
        else:
            self.head = nn.Linear(feature_dim, args.num_classes)
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out

class Head(nn.Module):
    def __init__(self, num_classes=10, hidden_dims=[512]):
        super().__init__()
        hidden_dims.append(num_classes)

        layers = []
        for idx in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[idx-1], hidden_dims[idx]))
            layers.append(nn.ReLU(inplace=True))

        self.fc = nn.Sequential(*layers)

    def forward(self, rep):
        out = self.fc(rep)
        return out

###########################################################

class CNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, height=28, 
                 num_cov=2, feature_dim=512, hidden_dims=[]):
        super().__init__()
        convs = [nn.Sequential(
                    nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
                    nn.ReLU(inplace=True), 
                    nn.MaxPool2d(kernel_size=(2, 2))
                )]
        height = int(height - 5 + 1)
        height = int((height - 2) / 2 + 1)
        i=-1
        for i in range(num_cov-1):
            convs.append(nn.Sequential(
                            nn.Conv2d(2**(i+5),
                                2**(i+6),
                                kernel_size=5,
                                padding=0,
                                stride=1,
                                bias=True),
                            nn.ReLU(inplace=True), 
                            nn.MaxPool2d(kernel_size=(2, 2))
                        ))
            height = int(height - 5 + 1)
            height = int((height - 2) / 2 + 1)
        self.conv = nn.Sequential(*convs)
        
        hidden_dims.append(feature_dim)

        layers = [nn.Flatten()]
        for idx in range(len(hidden_dims)):
            if len(layers) == 1:
                layers.append(nn.Linear(height ** 2 * 2**(i+6), hidden_dims[idx]))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Linear(hidden_dims[idx-1], hidden_dims[idx]))
                layers.append(nn.ReLU(inplace=True))

        self.fc1 = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.fc1(out)
        out = self.fc(out)
        return out

# https://github.com/jindongwang/Deep-learning-activity-recognition/blob/master/pytorch/network.py
class HARCNN(nn.Module):
    def __init__(self, in_channels=9, dim_hidden=64 * 26, num_classes=6, conv_kernel_size=(1, 9), stride=1, pool_kernel_size=(1, 2)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_hidden, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class HARCNN1(nn.Module):
    def __init__(self, in_channels=9, dim_hidden=32 * 26, num_classes=6, conv_kernel_size=(1, 9), stride=1, pool_kernel_size=(1, 2)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_hidden, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class HARCNN3(nn.Module):
    def __init__(self, in_channels=9, dim_hidden=128 * 26, num_classes=6, conv_kernel_size=(1, 5), stride=1, pool_kernel_size=(1, 2)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=conv_kernel_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(dim_hidden, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# https://github.com/FengHZ/KD3A/blob/master/model/digit5.py
class Digit5CNN(nn.Module):
    def __init__(self):
        super(Digit5CNN, self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module("conv1", nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn1", nn.BatchNorm2d(64))
        self.encoder.add_module("relu1", nn.ReLU())
        self.encoder.add_module("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.encoder.add_module("conv2", nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn2", nn.BatchNorm2d(64))
        self.encoder.add_module("relu2", nn.ReLU())
        self.encoder.add_module("maxpool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.encoder.add_module("conv3", nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn3", nn.BatchNorm2d(128))
        self.encoder.add_module("relu3", nn.ReLU())

        self.linear = nn.Sequential()
        self.linear.add_module("fc1", nn.Linear(8192, 3072))
        self.linear.add_module("bn4", nn.BatchNorm1d(3072))
        self.linear.add_module("relu4", nn.ReLU())
        self.linear.add_module("dropout", nn.Dropout())
        self.linear.add_module("fc2", nn.Linear(3072, 2048))
        self.linear.add_module("bn5", nn.BatchNorm1d(2048))
        self.linear.add_module("relu5", nn.ReLU())

        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        batch_size = x.size(0)
        feature = self.encoder(x)
        feature = feature.view(batch_size, -1)
        feature = self.linear(feature)
        out = self.fc(feature)
        return out
        

# https://github.com/FengHZ/KD3A/blob/master/model/amazon.py
class AmazonMLP(nn.Module):
    def __init__(self, feature_dim=[500]):
        super(AmazonMLP, self).__init__()
        self.in_features = 5000
        self.out_features = 100
        layers = []
        for idx in range(len(feature_dim)):
            if len(layers) == 0:
                layers.append(nn.Linear(self.in_features, feature_dim[idx]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(feature_dim[idx-1], feature_dim[idx]))
                layers.append(nn.ReLU())
        try:
            layers.append(nn.Linear(feature_dim[idx], self.out_features))
        except UnboundLocalError:
            layers.append(nn.Linear(self.in_features, self.out_features))
        layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(self.out_features, 2)

    def forward(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        return out
        

# # https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/cnn.py
# class FedAvgCNN(nn.Module):
#     def __init__(self, in_features=1, num_classes=10, dim=1024):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_features,
#                                32,
#                                kernel_size=5,
#                                padding=0,
#                                stride=1,
#                                bias=True)
#         self.conv2 = nn.Conv2d(32,
#                                64,
#                                kernel_size=5,
#                                padding=0,
#                                stride=1,
#                                bias=True)
#         self.fc1 = nn.Linear(dim, 512)
#         self.fc = nn.Linear(512, num_classes)

#         self.act = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

#     def forward(self, x):
#         x = self.act(self.conv1(x))
#         x = self.maxpool(x)
#         x = self.act(self.conv2(x))
#         x = self.maxpool(x)
#         x = torch.flatten(x, 1)
#         x = self.act(self.fc1(x))
#         x = self.fc(x)
#         return x

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

# ====================================================================================================================

# https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/mlp.py
class FedAvgMLP(nn.Module):
    def __init__(self, in_features=784, num_classes=10, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

# ====================================================================================================================

class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=1*28*28, num_classes=10):
        super(Mclr_Logistic, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

# ====================================================================================================================

class DNN(nn.Module):
    def __init__(self, input_dim=1*28*28, mid_dim=100, num_classes=10):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# ====================================================================================================================

# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGGbatch_size': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 10)
#         )

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         output = F.log_softmax(out, dim=1)
#         return output

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)

# ====================================================================================================================

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

class LeNet(nn.Module):
    def __init__(self, feature_dim=50*4*4, bottleneck_dim=256, num_classes=10, iswn=None):
        super(LeNet, self).__init__()

        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, num_classes)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# ====================================================================================================================

# class CNNCifar(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, batch_size, 5)
#         self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 100)
#         self.fc3 = nn.Linear(100, num_classes)

#         # self.weight_keys = [['fc1.weight', 'fc1.bias'],
#         #                     ['fc2.weight', 'fc2.bias'],
#         #                     ['fc3.weight', 'fc3.bias'],
#         #                     ['conv2.weight', 'conv2.bias'],
#         #                     ['conv1.weight', 'conv1.bias'],
#         #                     ]
                            
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, batch_size * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = F.log_softmax(x, dim=1)
#         return x

# ====================================================================================================================

class LSTMNet(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, bidirectional=False, dropout=0.2, 
                padding_idx=0, vocab_size=98635, num_classes=10):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout, 
                            batch_first=True)
        dims = hidden_dim*2 if bidirectional else hidden_dim
        self.fc = nn.Linear(dims, num_classes)

    def forward(self, x):
        if type(x) == type([]):
            text, text_lengths = x
        else:
            text, text_lengths = x, [x.shape[1] for _ in range(x.shape[0])]
        
        embedded = self.embedding(text)
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        #unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        out = torch.relu_(out[:,-1,:])
        out = self.dropout(out)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
            
        return out

# ====================================================================================================================

class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        if type(x) == type([]):
            text, _ = x
        else:
            text = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = F.log_softmax(z, dim=1)

        return out

# ====================================================================================================================

class TextCNN(nn.Module):
    def __init__(self, hidden_dim, num_channels=100, kernel_size=[3,4,5], max_len=200, dropout=0.8, 
                padding_idx=0, vocab_size=98635, num_classes=10):
        super(TextCNN, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # This stackoverflow thread clarifies how conv1d works
        # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[0]+1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[1]+1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[2]+1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Fully-Connected Layer
        self.fc1 = nn.Linear(num_channels*len(kernel_size), hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        if type(x) == type([]):
            text, _ = x
        else:
            text = x

        embedded_sent = self.embedding(text).permute(0,2,1)
        
        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)
        
        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        feat = self.fc1(final_feature_map)
        out = self.fc(feat)
        out = F.log_softmax(out, dim=1)

        return out

# ====================================================================================================================

class GRUNet(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, bidirectional=False, dropout=0.2, 
                 padding_idx=0, vocab_size=98635, num_classes=10):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.gru = nn.GRU(input_size=hidden_dim, 
                          hidden_size=hidden_dim, 
                          num_layers=num_layers, 
                          bidirectional=bidirectional, 
                          dropout=dropout, 
                          batch_first=True)
        dims = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(dims, num_classes)

    def forward(self, x):
        if isinstance(x, list):
            text, text_lengths = x
        else:
            text, text_lengths = x, [x.shape[1] for _ in range(x.shape[0])]

        embedded = self.embedding(text)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed_embedded)

        if isinstance(hidden, tuple):  # LSTM 返回 (hidden, cell)，GRU 只返回 hidden
            hidden = hidden[0]

        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        output = F.log_softmax(output, dim=1)

        return output


# ====================================================================================================================

class TextLogisticRegression(nn.Module):
    def __init__(self, hidden_dim, vocab_size=98635, num_classes=10):
        super(TextLogisticRegression, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if isinstance(x, list):
            text, _ = x
        else:
            text = x

        embedded = self.embedding(text)
        avg_embedding = embedded.mean(dim=1)  # 取句子 token 平均表示
        output = self.fc(avg_embedding)
        output = F.log_softmax(output, dim=1)

        return output

# ====================================================================================================================

# class linear(Function):
#   @staticmethod
#   def forward(ctx, input):
#     return input
  
#   @staticmethod
#   def backward(ctx, grad_output):
#     return grad_output

# ----------------------------SVD分解---------------------------------
#卷积层分解后的
class FactorizedConv(nn.Module):
    def __init__(self, in_channels, out_channels, rank_rate, padding=None, stride=1, kernel_size=3, bias=False):
        super(FactorizedConv, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding if padding is not None else 0
        self.groups = 1  # 分组卷积支持，默认为1

        # 计算低秩分解的秩
        # self.rank = max(1, round(rank_rate * min(in_channels, out_channels)))
        self.rank = max(1, round(rank_rate * min(out_channels * kernel_size, in_channels * kernel_size)))
        # 使用二维矩阵存储分解参数
        # 通用处理任意kernel_size
        self.dim1 = out_channels * kernel_size
        self.dim2 = in_channels * kernel_size

        # 低秩参数矩阵 (二维存储)

        self.conv_v = nn.Parameter(torch.Tensor(self.rank, self.dim2))
        self.conv_u = nn.Parameter(torch.Tensor(self.dim1, self.rank))

        # 偏置参数
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        #初始化模型参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv_u, a=math.sqrt(0))
        nn.init.kaiming_uniform_(self.conv_v, a=math.sqrt(0))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, x):
        # 空间分解: 1xK + Kx1
        # 垂直卷积 (1xK)
        weight_v = self.conv_v.T.reshape(self.in_channels, self.kernel_size, 1, self.rank).permute(3, 0, 2, 1)
        out = F.conv2d(
            x, weight_v, None,
            stride=(1, self.stride),
            padding=(0, self.padding),
            dilation=(1, 1),
            groups=self.groups
        )

        # 水平卷积 (Kx1)  不显示reshpe权重
        weight_u = self.conv_u.reshape(self.out_channels, self.kernel_size, self.rank, 1).permute(0, 2, 1, 3)
        out = F.conv2d(
            out, weight_u, self.bias,
            stride=(self.stride, 1),
            padding=(self.padding, 0),
            dilation=(1, 1),
            groups=self.groups
        )
        return out

    def frobenius_loss(self):
        return torch.sum((self.conv_u @ self.conv_v) ** 2)

    def reconstruct_full_weight(self):
        """重建完整的卷积核权重 (用于聚合)"""
        # 直接使用矩阵乘法
        A = self.conv_u @ self.conv_v  # [out*K, in*K]
        W = A.reshape(self.out_channels, self.kernel_size, self.in_channels, self.kernel_size)
        W = W.permute(0, 2, 1, 3)  # [out, in, K, K]
        return W

    def L2_loss(self):
        """分解参数的L2范数平方和"""
        return torch.norm(self.conv_u, p='fro') ** 2 + torch.norm(self.conv_v, p='fro') ** 2

    def kronecker_loss(self):
        """Kronecker乘积损失"""
        return (torch.norm(self.conv_u, p='fro') ** 2) * (torch.norm(self.conv_v, p='fro') ** 2)


# 卷积层分解函数 (FedHM兼容)
def Decom_COV(conv_model, ratio_LR=0.5):
    # 自动从卷积层获取参数
    in_planes = conv_model.in_channels
    out_planes = conv_model.out_channels
    kernel_size = conv_model.kernel_size[0]
    stride = conv_model.stride[0]
    padding = conv_model.padding[0]
    bias = conv_model.bias is not None

    # 创建分解层 (使用二维矩阵存储)
    factorized_cov = FactorizedConv(
        in_planes,
        out_planes,
        rank_rate=ratio_LR,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        bias=bias
    )

    # 获取原始权重并重塑
    W = conv_model.weight.data

    #重塑: [out, in, K, K] -> [out*K, in*K]
    A = W.permute(0, 2, 1, 3).reshape(out_planes * kernel_size, in_planes * kernel_size)

    # SVD分解
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    # 计算截断秩
    rank = factorized_cov.rank
    S_sqrt = torch.sqrt(S[:rank])
    # 分配奇异值
    U_weight = U[:, :rank] @ torch.diag(S_sqrt)
    V_weight = torch.diag(S_sqrt) @ Vh[:rank, :]

    # 加载参数
    with torch.no_grad():
        factorized_cov.conv_u.copy_(U_weight)
        factorized_cov.conv_v.copy_(V_weight)

        # 复制偏置
        if bias:
            factorized_cov.bias.copy_(conv_model.bias.data)

    return factorized_cov

# 卷积层恢复函数
def Recover_COV(decom_conv):
    # 获取分解层参数
    in_planes = decom_conv.in_channels
    out_planes = decom_conv.out_channels
    kernel_size = decom_conv.kernel_size
    stride = decom_conv.stride
    padding = decom_conv.padding
    bias = decom_conv.bias is not None

    # 重建完整权重
    W = decom_conv.reconstruct_full_weight()

    # 创建原始卷积层
    recovered_conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=kernel_size,
        stride=stride, padding=padding, bias=bias
    )

    # 加载权重
    with torch.no_grad():
        recovered_conv.weight.copy_(W)
        if bias:
            recovered_conv.bias.copy_(decom_conv.bias)

    return recovered_conv

# 分解后的全连接层 (二维矩阵存储)
class FactorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank_rate, bias=True):
        super(FactorizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #中间rank值
        self.rank = max(1, round(rank_rate * min(in_features, out_features)))

        # 二维矩阵参数
        #第一个全连接层的参数（维度为 r*in）
        self.weight_v = nn.Parameter(torch.Tensor(self.rank, in_features))
        #第二个全连接层的参数 (维度为 out*r)
        self.weight_u = nn.Parameter(torch.Tensor(out_features, self.rank))

        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_u, a=math.sqrt(0))
        nn.init.kaiming_uniform_(self.weight_v, a=math.sqrt(0))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight_u.size(1))  # rank
            nn.init.uniform_(self.bias, -bound, bound)
    def reconstruct_full_weight(self):
        return self.weight_u @ self.weight_v
        
    def forward(self, x):
        """
        前向传播分为两步：
        1. 降维投影：x -> (batch,  rank)
        2. 升维投影：x -> (batch, out_features)
        """
        # 第一步：降维投影 (输入特征空间 -> 低秩空间)  传入 F.linear() 的权重存储形式与正常线性层 (nn.Linear) 完全一致必须是（out*in）
        x = F.linear(x, self.weight_v, None)  # 形状: (batch,  rank)

        # 第二步：升维投影 (低秩空间 -> 输出特征空间)
        x = F.linear(x, self.weight_u, self.bias)  # 形状: (batch, out_features)

        return x

    def frobenius_loss(self):
        W = self.weight_u @ self.weight_v
        return torch.sum(W ** 2)

    def L2_loss(self):
        return torch.norm(self.weight_v) ** 2 + torch.norm(self.weight_u) ** 2

    def kronecker_loss(self):
        return (torch.norm(self.weight_v) ** 2) * (torch.norm(self.weight_u) ** 2)


# 全连接层分解函数(将全连接权重  W（out*in）分解为 out*r（第二个全连接权重） r*in （第一个权全连接重）)
def Decom_LINEAR(linear_model, ratio_LR=0.5):
    in_features = linear_model.in_features
    out_features = linear_model.out_features
    has_bias = linear_model.bias is not None

    # 创建分解层
    factorized_linear = FactorizedLinear(in_features, out_features, ratio_LR, has_bias)

    # SVD分解（属注意与torch.svd函数的区别  主要是第三个矩阵）  Vh是V矩阵的转置（就是第三个矩阵）
    U, S, Vh = torch.linalg.svd(linear_model.weight.data, full_matrices=False)

    # 计算截断秩
    rank = factorized_linear.rank

    # 分配奇异值  第一个矩阵切列   第三个矩阵切行
    S_sqrt = torch.sqrt(S[:rank])
    U_weight = U[:, :rank] @ torch.diag(S_sqrt)  #shape out*r
    V_weight = torch.diag(S_sqrt) @ Vh[:rank, :]  #shape r*in

    # 加载参数
    with torch.no_grad():
        factorized_linear.weight_v.copy_(V_weight)
        factorized_linear.weight_u.copy_(U_weight)
        if has_bias:
            factorized_linear.bias.copy_(linear_model.bias.data)

    return factorized_linear


# 全连接层恢复函数
def Recover_LINEAR(factorized_linear):
    in_features = factorized_linear.in_features
    out_features = factorized_linear.out_features
    has_bias = factorized_linear.bias is not None

    # # 重建权重
    # weight = factorized_linear.weight_u @ factorized_linear.weight_v
    weight = factorized_linear.reconstruct_full_weight()

    # 创建原始线性层
    recovered_linear = nn.Linear(in_features, out_features, bias=has_bias)

    # 加载参数
    with torch.no_grad():
        recovered_linear.weight.copy_(weight)
        if has_bias:
            recovered_linear.bias.copy_(factorized_linear.bias)

    return recovered_linear

# ARA DAR用的模型
class Hyper_CNN(nn.Module):
    def __init__(self, in_features=3, num_classes=10, n_kernels=16, ratio_LR=0.7):
        super(Hyper_CNN, self).__init__()
        self.ratio_LR = ratio_LR

        # 卷积层和池化层
        self.conv1 = nn.Conv2d(in_features, n_kernels, 5)
        if ratio_LR >= 1.0:
            self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        else:
            self.conv2 = FactorizedConv(in_channels=n_kernels, out_channels=2* n_kernels, padding=0,rank_rate=ratio_LR,kernel_size=5,bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # 计算全连接层输入维度
        self.fc_input_dim = 2 * n_kernels * 5 * 5
        # #
        # print(f"全连接层的输入维度为：",self.fc_input_dim)
        # 全连接层（可能低秩分解）
        if ratio_LR >= 1.0:
            self.fc1 = nn.Linear(self.fc_input_dim, 2000)
            self.fc2 = nn.Linear(2000, 500)
        else:
            self.fc1 = FactorizedLinear(in_features=self.fc_input_dim, out_features=2000, rank_rate=ratio_LR, bias=True)
            self.fc2 = FactorizedLinear(2000, 500, rank_rate=ratio_LR, bias=True)

        # 激活函数和输出层
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(500, num_classes)

        # 构建共享基础部分
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
            self.relu
        )

        # 个性化头部
        self.head = self.fc3

    def recover_larger_model(self):
        """将低秩层恢复为完整秩"""
        if self.ratio_LR >= 1.0:
            return
        self.conv2 = Recover_COV(self.conv2)
        # 恢复两个全连接层
        self.fc1 = Recover_LINEAR(self.fc1)
        self.fc2 = Recover_LINEAR(self.fc2)
        # 更新base索引
        self._rebuild_base()
        print("(卷积)恢复低秩模型为完整模型，fc1和fc2已恢复")

    def decom_larger_model(self, rank_rate):
        """将完整秩层分解为低秩"""
        if rank_rate >= 1.0:
            return

        if isinstance(self.conv2,nn.Conv2d):
            self.conv2 = Decom_COV(self.conv2,rank_rate)
        if isinstance(self.fc1, nn.Linear):
            self.fc1 = Decom_LINEAR(self.fc1, rank_rate)
        if isinstance(self.fc2, nn.Linear):
            self.fc2 = Decom_LINEAR(self.fc2, rank_rate)

        self._rebuild_base()
        print(f"将完整模型分解(卷积也分解)为低秩模型(rank_rate={rank_rate})")

    def _rebuild_base(self):
        """重构基础网络部分"""
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
            self.relu
        )

    def frobenius_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.frobenius_loss()+self.fc2.frobenius_loss() + self.conv2.frobenius_loss()

    def kronecker_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.kronecker_loss()+self.fc2.kronecker_loss() + self.conv2.kronecker_loss()

    def L2_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.L2_loss() +self.fc2.L2_loss()+ self.conv2.L2_loss()

    def forward(self, x):
        features = self.base(x)  # 提取特征
        output = self.head(features)  # 分类输出
        return output


# 不分解卷积层的低秩CNN网络
# class Hyper_CNN(nn.Module):
#     def __init__(self, in_features=3, num_classes=10, n_kernels=16, ratio_LR=0.7):
#         super(Hyper_CNN, self).__init__()
#         self.ratio_LR = ratio_LR

#         # 卷积层和池化层
#         self.conv1 = nn.Conv2d(in_features, n_kernels, 5)
#         self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.flatten = nn.Flatten()

#         # 计算全连接层输入维度
#         self.fc_input_dim = 2 * n_kernels * 5 * 5
#         # #
#         # print(f"全连接层的输入维度为：",self.fc_input_dim)
#         # 全连接层（可能低秩分解）
#         if ratio_LR >= 1.0:
#             self.fc1 = nn.Linear(self.fc_input_dim, 2000)
#             self.fc2 = nn.Linear(2000, 500)
#         else:
#             self.fc1 = FactorizedLinear(in_features=self.fc_input_dim, out_features=2000, rank_rate=ratio_LR, bias=True)
#             self.fc2 = FactorizedLinear(2000, 500, rank_rate=ratio_LR, bias=True)

#         # 激活函数和输出层
#         self.relu = nn.ReLU()
#         self.fc3 = nn.Linear(500, num_classes)

#         # 构建共享基础部分
#         self.base = nn.Sequential(
#             self.conv1,
#             self.relu,
#             self.pool,
#             self.conv2,
#             self.relu,
#             self.pool,
#             self.flatten,
#             self.fc1,
#             self.relu,
#             self.fc2,
#             self.relu
#         )

#         # 个性化头部
#         self.head = self.fc3

#     def recover_larger_model(self):
#         """将低秩层恢复为完整秩"""
#         if self.ratio_LR >= 1.0:
#             return
#         # 恢复两个全连接层
#         self.fc1 = Recover_LINEAR(self.fc1)
#         self.fc2 = Recover_LINEAR(self.fc2)
#         # 更新base索引
#         self._rebuild_base()
#         print("恢复低秩模型为完整模型，fc1和fc2已恢复")

#     def decom_larger_model(self, rank_rate):
#         """将完整秩层分解为低秩"""
#         if rank_rate >= 1.0:
#             return
#         if isinstance(self.fc1, nn.Linear):
#             self.fc1 = Decom_LINEAR(self.fc1, rank_rate)
#         if isinstance(self.fc2, nn.Linear):
#             self.fc2 = Decom_LINEAR(self.fc2, rank_rate)

#         self._rebuild_base()
#         print(f"将完整模型分解为低秩模型(rank_rate={rank_rate})")

#     def _rebuild_base(self):
#         """动态重构基础网络部分"""
#         self.base = nn.Sequential(
#             self.conv1,
#             self.relu,
#             self.pool,
#             self.conv2,
#             self.relu,
#             self.pool,
#             self.flatten,
#             self.fc1,
#             self.relu,
#             self.fc2,
#             self.relu
#         )

#     def frobenius_decay(self):
#         if self.ratio_LR >= 1.0:
#             return torch.tensor(0., device=self.conv1.weight.device)
#         return self.fc1.frobenius_loss()+self.fc2.frobenius_loss() 
#     def kronecker_decay(self):
#         if self.ratio_LR >= 1.0:
#             return torch.tensor(0., device=self.conv1.weight.device)
#         return self.fc1.kronecker_loss()+self.fc2.kronecker_loss() 

#     def L2_decay(self):
#         if self.ratio_LR >= 1.0:
#             return torch.tensor(0., device=self.conv1.weight.device)
#         return self.fc1.L2_loss() +self.fc2.L2_loss()

#     def forward(self, x):
#         features = self.base(x)  # 提取特征
#         output = self.head(features)  # 分类输出
#         return output


# [0.1,0.10,0.20,0.25,0.35]
# [0.05,0.10,0.10,0.35,0.40]
# [0.05,0.05,0.05,0.15,0.7]
# def Model_Distribe(args, cid=0, is_global=False, ratios=[0.05,0.05,0.05,0.15,0.7], num_clients=20):
#     if is_global:
#         # 服务器模型
#         model = eval(args.global_model)
#         return model

#     assert ratios is not None, "ratios must be provided for client model distribution"
#     assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
#     assert len(ratios) == len(args.models), "ratios length must match models length"

#     # 计算每个模型分配的客户端数量
#     client_nums = [int(r * num_clients) for r in ratios]

#     # 处理由于取整导致的剩余客户端
#     remainder = num_clients - sum(client_nums)
#     for i in range(remainder):
#         client_nums[i] += 1  # 把剩余的补到前几个模型

#     # 计算 cid 对应的模型索引
#     cum_sum = 0
#     model_idx = None
#     for i, n in enumerate(client_nums):
#         cum_sum += n
#         if cid < cum_sum:
#             model_idx = i
#             break

#     assert model_idx is not None

#     model = eval(args.models[model_idx])
#     return model

    
    
#重写这个需要使用    
def Model_Distribe(args, cid=0, is_global=False):
    if is_global:
    #服务器模型创建和分割成特征提取器和分类器
        model = eval(args.global_model)
    else:
        #客户端创建和分割特征提取器和分类器
        model = eval(args.models[cid % len(args.models)])
    return model

#不低秩分解的几个CNN模型
class BaseHeadCNN(nn.Module):
    """基础类，包含共享的base-head结构"""
    def __init__(self, base, head):
        super().__init__()
        self.base = base
        self.head = head
        
    def forward(self, x):
        features = self.base(x)
        return self.head(features)

# 原始CNN_1重构
class CNN_1(BaseHeadCNN):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        # 基础部分
        base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU()
        )
        
        # 分类器头
        head = nn.Linear(500, out_dim)
        
        super().__init__(base, head)

# 原始CNN_2重构
class CNN_2(BaseHeadCNN):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(n_kernels * 5 * 5, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU()
        )
        
        head = nn.Linear(500, out_dim)
        super().__init__(base, head)

# 原始CNN_3重构
class CNN_3(BaseHeadCNN):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU()
        )
        
        head = nn.Linear(500, out_dim)
        super().__init__(base, head)

# 原始CNN_4重构
class CNN_4(BaseHeadCNN):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 800),
            nn.ReLU(),
            nn.Linear(800, 500),
            nn.ReLU()
        )
        
        head = nn.Linear(500, out_dim)
        super().__init__(base, head)

# 原始CNN_5重构
class CNN_5(BaseHeadCNN):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU()
        )
        
        head = nn.Linear(500, out_dim)
        super().__init__(base, head)

# -------------------------------------------------FedAFM

class CNN_1_hetero_AFM(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_1_hetero_AFM, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU()
        )

        self.head = nn.Linear(500, out_dim)
        

    def forward(self, x, homo_rep, alpha):
        feature = self.base(x)
        mix_feature = feature*alpha.to(homo_rep.device) + homo_rep
        output = self.head(mix_feature)
        return output,mix_feature

class CNN_2_hetero_AFM(nn.Module): # change filters of convs
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_2_hetero_AFM, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(n_kernels * 5 * 5, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU()
        )
        
        self.head = nn.Linear(500, out_dim)

    def forward(self, x, homo_rep, alpha):
        feature = self.base(x)
        mix_feature = feature*alpha.to(homo_rep.device) + homo_rep
        output = self.head(mix_feature)
        return output,mix_feature

class CNN_3_hetero_AFM(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_3_hetero_AFM, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU()
        )
        
        self.head = nn.Linear(500, out_dim)

    def forward(self, x, homo_rep, alpha):
        feature = self.base(x)
        mix_feature = feature*alpha.to(homo_rep.device) + homo_rep
        output = self.head(mix_feature)
        return output,mix_feature


class CNN_4_hetero_AFM(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_4_hetero_AFM, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 800),
            nn.ReLU(),
            nn.Linear(800, 500),
            nn.ReLU()
        )
        
        self.head = nn.Linear(500, out_dim)

    def forward(self, x, homo_rep, alpha):
        feature = self.base(x)
        mix_feature = feature*alpha.to(homo_rep.device) + homo_rep
        output = self.head(mix_feature)
        return output,mix_feature

class CNN_5_hetero_AFM(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_5_hetero_AFM, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU()
        )
        
        self.head = nn.Linear(500, out_dim)

    def forward(self, x, homo_rep, alpha):
        feature = self.base(x)
        mix_feature = feature*alpha.to(homo_rep.device) + homo_rep
        output = self.head(mix_feature)
        return output,mix_feature

class CNN_5_homo_AFM(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_5_homo_AFM, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU()
        )
        
        self.head = nn.Linear(500, out_dim)

    def forward(self, x):
        feature = self.base(x)
        output = self.head(feature)
        return output, feature


#afm的可训练向量,计算（1-alpha）*同构表征
class vector_alpha(nn.Module):
    def __init__(self):
        super(vector_alpha, self).__init__()

        self.alpha = nn.Parameter(torch.ones(512))

    def forward(self, small_input):
        output = (1-self.alpha.to(small_input.device)) * small_input
        return output



#其他基线使用的同构模型
class Decom_FedAvgCNN(nn.Module):
    def __init__(self, in_features=3, num_classes=10, dim=1600, ratio_LR=0.7):
        super(Decom_FedAvgCNN, self).__init__()
        self.ratio_LR = ratio_LR

        # 卷积层和池化层
        self.conv1 = nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True)

        self.conv2 = nn.Conv2d(32,
                               64,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        # 全连接层（可能低秩分解）
        if ratio_LR >= 1.0:
            self.fc1 = nn.Linear(dim, 512)
        else:
            self.fc1 = FactorizedLinear(in_features=dim, out_features=512, rank_rate=ratio_LR, bias=True)

        fc2 = nn.Linear(512, num_classes)

        # 构建共享基础部分
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu
        )

        # 个性化头部
        self.head = fc2

    def recover_larger_model(self):
        """将低秩层恢复为完整秩"""
        if self.ratio_LR >= 1.0:
            return
        self.fc1 = Recover_LINEAR(self.fc1)
        # 更新base索引
        self._rebuild_base()
        print("恢复低秩模型为完整模型，fc1已恢复")

    def decom_larger_model(self, rank_rate):
        """将完整秩层分解为低秩"""
        if rank_rate >= 1.0:
            return

        if isinstance(self.fc1, nn.Linear):
            self.fc1 = Decom_LINEAR(self.fc1, rank_rate)

        self._rebuild_base()
        print(f"将完整模型分解为低秩模型(rank_rate={rank_rate})")

    def _rebuild_base(self):
        """重构基础网络部分"""
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu
        )

    def frobenius_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.frobenius_loss()

    def kronecker_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.kronecker_loss()

    def L2_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.L2_loss()

    def forward(self, x):
        features = self.base(x)  # 提取特征
        output = self.head(features)  # 分类输出
        return output
#pFedAFM专用同构(异构大模型) 
class FedAvgCNN_Hetero_AFM(nn.Module):
    def __init__(self, in_features=3, num_classes=10, dim=1600, ratio_LR=0.7):
        super(FedAvgCNN_Hetero_AFM, self).__init__()
        self.ratio_LR = ratio_LR

        # 卷积层和池化层
        self.conv1 = nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True)

        self.conv2 = nn.Conv2d(32,
                               64,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        # 全连接层（可能低秩分解）
        if ratio_LR >= 1.0:
            self.fc1 = nn.Linear(dim, 512)
        else:
            self.fc1 = FactorizedLinear(in_features=dim, out_features=512, rank_rate=ratio_LR, bias=True)

        fc2 = nn.Linear(512, num_classes)

        # 构建共享基础部分
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu
        )

        # 个性化头部
        self.head = fc2

    def recover_larger_model(self):
        """将低秩层恢复为完整秩"""
        if self.ratio_LR >= 1.0:
            return
        self.fc1 = Recover_LINEAR(self.fc1)
        # 更新base索引
        self._rebuild_base()
        print("恢复低秩模型为完整模型，fc1已恢复")

    def decom_larger_model(self, rank_rate):
        """将完整秩层分解为低秩"""
        if rank_rate >= 1.0:
            return

        if isinstance(self.fc1, nn.Linear):
            self.fc1 = Decom_LINEAR(self.fc1, rank_rate)

        self._rebuild_base()
        print(f"将完整模型分解为低秩模型(rank_rate={rank_rate})")

    def _rebuild_base(self):
        """重构基础网络部分"""
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu
        )

    def frobenius_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.frobenius_loss()

    def kronecker_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.kronecker_loss()

    def L2_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.L2_loss()

    def forward(self, x, homo_rep, alpha):
        features = self.base(x)  # 提取特征
        mix_features = features*alpha.to(homo_rep.device) + homo_rep
        output = self.head(features)  # 分类输出
        return output,mix_features
    
class FedAvgCNN_Homo_AFM(nn.Module):
    def __init__(self, in_features=3, num_classes=10, dim=1600, ratio_LR=0.7):
        super(FedAvgCNN_Homo_AFM, self).__init__()
        self.ratio_LR = ratio_LR

        # 卷积层和池化层
        self.conv1 = nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True)

        self.conv2 = nn.Conv2d(32,
                               64,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        # 全连接层（可能低秩分解）
        if ratio_LR >= 1.0:
            self.fc1 = nn.Linear(dim, 512)
        else:
            self.fc1 = FactorizedLinear(in_features=dim, out_features=512, rank_rate=ratio_LR, bias=True)

        fc2 = nn.Linear(512, num_classes)

        # 构建共享基础部分
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu
        )

        # 个性化头部
        self.head = fc2

    def recover_larger_model(self):
        """将低秩层恢复为完整秩"""
        if self.ratio_LR >= 1.0:
            return
        self.fc1 = Recover_LINEAR(self.fc1)
        # 更新base索引
        self._rebuild_base()
        print("恢复低秩模型为完整模型，fc1已恢复")

    def decom_larger_model(self, rank_rate):
        """将完整秩层分解为低秩"""
        if rank_rate >= 1.0:
            return

        if isinstance(self.fc1, nn.Linear):
            self.fc1 = Decom_LINEAR(self.fc1, rank_rate)

        self._rebuild_base()
        print(f"将完整模型分解为低秩模型(rank_rate={rank_rate})")

    def _rebuild_base(self):
        """重构基础网络部分"""
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu
        )

    def frobenius_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.frobenius_loss()

    def kronecker_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.kronecker_loss()

    def L2_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.L2_loss()

    def forward(self, x):
        features = self.base(x)  # 提取特征
        output = self.head(features)  # 分类输出
        return output,features
# afm的可训练向量,计算（1-alpha）*同构表征
# class vector_alpha(nn.Module):
#     def __init__(self):
#         super(vector_alpha, self).__init__()

#         self.alpha = nn.Parameter(torch.ones(512))

#     def forward(self, small_input):
#         output = (1-self.alpha.to(small_input.device)) * small_input
#         return output

# ---------------------------TinyImageNet使用的模型结构--------------------------------------------
class CNN_1_tiny(BaseHeadCNN):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        # 基础部分
        base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(5408, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU()
        )

        # 分类器头
        head = nn.Linear(500, out_dim)

        super().__init__(base, head)


class CNN_2_tiny(BaseHeadCNN):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2704, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU()
        )

        head = nn.Linear(500, out_dim)
        super().__init__(base, head)

class CNN_3_tiny(BaseHeadCNN):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(5408, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU()
        )

        head = nn.Linear(500, out_dim)
        super().__init__(base, head)

class CNN_4_tiny(BaseHeadCNN):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(5408, 800),
            nn.ReLU(),
            nn.Linear(800, 500),
            nn.ReLU()
        )

        head = nn.Linear(500, out_dim)
        super().__init__(base, head)

class CNN_5_tiny(BaseHeadCNN):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(5408, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU()
        )

        head = nn.Linear(500, out_dim)
        super().__init__(base, head)
        
        

#卷积层相较于全连接参数太少了不进行分解（分解也会严重影响性能）
class Hyper_CNN_tiny(nn.Module):
    def __init__(self, in_features=3, num_classes=10, n_kernels=16, ratio_LR=0.7):
        super(Hyper_CNN_tiny, self).__init__()
        self.ratio_LR = ratio_LR

        # 卷积层和池化层
        self.conv1 = nn.Conv2d(in_features, n_kernels, 5)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # 计算全连接层输入维度
        self.fc_input_dim = 5408
        # #
        # print(f"全连接层的输入维度为：",self.fc_input_dim)
        # 全连接层（可能低秩分解）
        if ratio_LR >= 1.0:
            self.fc1 = nn.Linear(self.fc_input_dim, 2000)
            self.fc2 = nn.Linear(2000, 500)
        else:
            self.fc1 = FactorizedLinear(in_features=self.fc_input_dim, out_features=2000, rank_rate=ratio_LR, bias=True)
            self.fc2 = FactorizedLinear(2000, 500, rank_rate=ratio_LR, bias=True)

        # 激活函数和输出层
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(500, num_classes)

        # 构建共享基础部分
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
            self.relu
        )

        # 个性化头部
        self.head = self.fc3

    def recover_larger_model(self):
        """将低秩层恢复为完整秩"""
        if self.ratio_LR >= 1.0:
            return
        # 恢复两个全连接层
        self.fc1 = Recover_LINEAR(self.fc1)
        self.fc2 = Recover_LINEAR(self.fc2)
        # 更新base索引
        self._rebuild_base()
        print("恢复低秩模型为完整模型，fc1和fc2已恢复")

    def decom_larger_model(self, rank_rate):
        """将完整秩层分解为低秩"""
        if rank_rate >= 1.0:
            return
        if isinstance(self.fc1, nn.Linear):
            self.fc1 = Decom_LINEAR(self.fc1, rank_rate)
        if isinstance(self.fc2, nn.Linear):
            self.fc2 = Decom_LINEAR(self.fc2, rank_rate)

        self._rebuild_base()
        print(f"将完整模型分解为低秩模型(rank_rate={rank_rate})")

    def _rebuild_base(self):
        """动态重构基础网络部分"""
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
            self.relu
        )

    def frobenius_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0., device=self.conv1.weight.device)
        return self.fc1.frobenius_loss()+self.fc2.frobenius_loss()
    def kronecker_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0., device=self.conv1.weight.device)
        return self.fc1.kronecker_loss()+self.fc2.kronecker_loss()

    def L2_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0., device=self.conv1.weight.device)
        return self.fc1.L2_loss() +self.fc2.L2_loss()

    def forward(self, x):
        features = self.base(x)  # 提取特征
        output = self.head(features)  # 分类输出
        return output
 
# -----------------------------------------------AFM
class CNN_1_hetero_AFM_tiny(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_1_hetero_AFM_tiny, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(5408, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU()
        )

        self.head = nn.Linear(500, out_dim)
        

    def forward(self, x, homo_rep, alpha):
        feature = self.base(x)
        mix_feature = feature*alpha.to(homo_rep.device) + homo_rep
        output = self.head(mix_feature)
        return output,mix_feature

class CNN_2_hetero_AFM_tiny(nn.Module): # change filters of convs
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_2_hetero_AFM_tiny, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2704, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU()
        )
        
        self.head = nn.Linear(500, out_dim)

    def forward(self, x, homo_rep, alpha):
        feature = self.base(x)
        mix_feature = feature*alpha.to(homo_rep.device) + homo_rep
        output = self.head(mix_feature)
        return output,mix_feature

class CNN_3_hetero_AFM_tiny(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_3_hetero_AFM_tiny, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(5408, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU()
        )
        
        self.head = nn.Linear(500, out_dim)

    def forward(self, x, homo_rep, alpha):
        feature = self.base(x)
        mix_feature = feature*alpha.to(homo_rep.device) + homo_rep
        output = self.head(mix_feature)
        return output,mix_feature


class CNN_4_hetero_AFM_tiny(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_4_hetero_AFM_tiny, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(5408, 800),
            nn.ReLU(),
            nn.Linear(800, 500),
            nn.ReLU()
        )
        
        self.head = nn.Linear(500, out_dim)

    def forward(self, x, homo_rep, alpha):
        feature = self.base(x)
        mix_feature = feature*alpha.to(homo_rep.device) + homo_rep
        output = self.head(mix_feature)
        return output,mix_feature

class CNN_5_hetero_AFM_tiny(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_5_hetero_AFM_tiny, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(5408, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU()
        )
        
        self.head = nn.Linear(500, out_dim)

    def forward(self, x, homo_rep, alpha):
        feature = self.base(x)
        mix_feature = feature*alpha.to(homo_rep.device) + homo_rep
        output = self.head(mix_feature)
        return output,mix_feature

class CNN_5_homo_AFM_tiny(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_5_homo_AFM_tiny, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(5408, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU()
        )
        
        self.head = nn.Linear(500, out_dim)

    def forward(self, x):
        feature = self.base(x)
        output = self.head(feature)
        return output, feature

# ----------------------------------------------TT分解---------------------------------------------------------------------
# ---------- 辅助函数 ----------
def Factorize(num):
    """原来的最接近两个因子分解（保留）"""
    if num <= 1:
        return [1, num]
    root = int(math.isqrt(num))
    for i in range(root, 0, -1):
        if num % i == 0:
            return [i, num // i]
    return [1, num]


def Decom_TT_COV(conv_model, ratio_LR=0.5):
    # 自动从卷积层获取参数
    in_planes = conv_model.in_channels
    out_planes = conv_model.out_channels
    kernel_size = conv_model.kernel_size[0]
    stride = conv_model.stride[0]
    padding = conv_model.padding[0]
    bias = conv_model.bias is not None

    # 创建分解层 (使用二维矩阵存储)
    factorized_cov = FactorizedConv(
        in_planes,
        out_planes,
        rank_rate=ratio_LR,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        bias=bias
    )

    # 获取原始权重并重塑
    W = conv_model.weight.data.clone()

    # 重塑: [out, in, K, K] -> [out*K, in*K]
    A = W.permute(0, 2, 1, 3).reshape(out_planes * kernel_size, in_planes * kernel_size)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    # 计算截断秩
    rank = factorized_cov.rank
    # 分配奇异值（第一种分配方式）
    S_sqrt = torch.sqrt(S[:rank])
    U_weight = U[:, :rank] @ torch.diag(S_sqrt)
    V_weight = torch.diag(S_sqrt) @ Vh[:rank, :]
    #第二种分配方式
    # U_weight = U[:, :rank]
    # V_weight = torch.diag(S[:rank]) @ Vh[:rank, :]

    # 加载参数
    with torch.no_grad():
        factorized_cov.conv_u.copy_(U_weight)
        factorized_cov.conv_v.copy_(V_weight)

        # 复制偏置
        if bias:
            factorized_cov.bias.copy_(conv_model.bias.data)

    return factorized_cov


# ---------- TT-SVD ----------
def tt_svd(W_reshaped, tt_ranks_rate, device=None, dtype=None):
    """
    通用的 TT-SVD（逐模分解）实现。
    - W_reshaped: tensor，shape = (d1, d2, ..., dk)
    - tt_ranks: list of length k-1，表示每一段的秩 (r1, r2, ..., r_{k-1})
    返回 cores 列表，cores[i] 形状为 (r_{i-1}, d_i, r_i) （r_0 = 1, r_k = 1）
    """
    if device is None:
        device = W_reshaped.device
    if dtype is None:
        dtype = W_reshaped.dtype

    shape = list(W_reshaped.shape)
    k = len(shape)

    cores = []
    # C 用于保存当前剩余张量（矩阵化）
    C = W_reshaped.clone().to(device=device, dtype=dtype)
    # 初始把 C reshape 成 (d1, d2*d3*...*dk)
    C = C.reshape(shape[0], -1)
    r_pre = 1
    for i in range(k - 1):
        # SVD
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        # 使用矩阵的两个维度确定rank
        max_rank = round(min(C.size(0), C.size(1)) * tt_ranks_rate)
        r = max(1, max_rank)
        # print(f"该层中间设置的秩为{r}")
        # print(f"核心截断的rank为{r}")
        U_trunc = U[:, :r]  # (left_dim, r)
        S_trunc = S[:r]  # (r,)
        Vh_trunc = Vh[:r, :]  # (r, right_dim)

        # 将奇异值分给分给U和V（第一种分配方式）
        S_trunc_sqrt = torch.sqrt(S_trunc)
        U_weight = U_trunc@ torch.diag(S_trunc_sqrt)
        core = U_weight.reshape(r_pre, shape[i], r).contiguous()
        cores.append(core.to(device=device, dtype=dtype))
        #迭代更新上一个r
        r_pre = r
        # 更新 C = S_trunc_sqrt @ Vh_trunc  -> (r, right_dim)
        C = torch.diag(S_trunc_sqrt) @ Vh_trunc

        # # 第二种分配方式
        # core = U_trunc.reshape(r_pre, shape[i], r).contiguous()
        # cores.append(core.to(device=device, dtype=dtype))
        # # 迭代更新上一个r
        # r_pre = r
        # # 更新 C = S_trunc @ Vh_trunc  -> (r, right_dim)
        # C = torch.diag(S_trunc) @ Vh_trunc
        # 如果后续还有维度，要 reshape 为 (r * d_{i+1}, rest)
        if i < k - 2:
            next_block = shape[i + 1]
            C = C.reshape(r * next_block, -1)

    # 最后一个 core: r_{k-2} × d_k × 1
    last_core = C.reshape(r_pre, shape[-1], 1).contiguous()
    cores.append(last_core.to(device=device, dtype=dtype))
    return cores



class TTLinear(nn.Module):
    """
    通用的 TT-分解的全连接层，支持任意数量的核心
    当只有两个核心时，自动等价于矩阵分解 W = U @ V
    """

    def __init__(self, in_features, out_features,
                 in_dims=None, out_dims=None,
                 tt_rank_rate=0.5, bias=True,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        # # # 默认把输入分成两个因子（原来行为）
        # if in_dims is None:
        #     in_dims = [in_features]
        # if out_dims is None:
        #     out_dims = [out_features]

        # # # 维度分解
        if in_dims is None:
            # 默认把输入分成四个因子（原来行为）
            in_dims = Factorize(in_features)
        if out_dims is None:
            out_dims = Factorize(out_features)

        # if in_dims is None and out_dims is None:
        #     if in_features > out_features:
        #         in_dims = Factorize(in_features)
        #         out_dims = [out_features]
        #     else:
        #         in_dims = [in_features]
        #         out_dims = Factorize(out_features)

        assert math.prod(in_dims) == in_features, f"in_dims乘积应等于in_features"
        assert math.prod(out_dims) == out_features, f"out_dims乘积应等于out_features"

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.p = len(in_dims)  # 输入核心数
        self.q = len(out_dims)  # 输出核心数
        self.k = self.p + self.q  # 总核心数
        # print(f"总核心数为{self.k}")
        # 创建TT核心
        self.tt_cores = nn.ParameterList()
        dims = self.out_dims + self.in_dims  # 注意：输出维度在前，输入维度在后

        # 计算秩
        ranks = [1]
        for i in range(self.k - 1):
            # 计算最大可能秩（基于矩阵秩的上界）
            # max_rank = min(
            #     math.prod(dims[:i+1]),  # 左边所有维度的乘积
            #     math.prod(dims[i+1:])   # 右边所有维度的乘积
            # )
            rank_1 = round(min(dims[i] * ranks[-1], math.prod(dims[i + 1:])) * tt_rank_rate)
            # 使用rank_rate计算实际秩
            rank = max(1, rank_1)
            ranks.append(rank)
        ranks.append(1)

        # 创建核心
        for i in range(self.k):
            r_prev = ranks[i]
            d_i = dims[i]
            r_next = ranks[i + 1]
            core = nn.Parameter(torch.empty(r_prev, d_i, r_next,
                                            device=device, dtype=dtype))
            self.tt_cores.append(core)

        # 偏置
        self.bias_flag = bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features,
                                                 device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        # 初始化
        self.reset_parameters()

    # def reset_parameters(self):
    #     """初始化TT核心和偏置"""
    #     for i, core in enumerate(self.tt_cores):
    #         # 使用正交初始化以获得更好的数值稳定性
    #         if i < self.q:  # 输出核心
    #             # 对应U矩阵的部分，使用正交初始化
    #             with torch.no_grad():
    #                 if core.size(0) == 1:  # 第一个核心
    #                     nn.init.orthogonal_(core.squeeze(0))
    #                 else:
    #                     # 对于非第一个核心，使用Kaiming初始化
    #                     nn.init.kaiming_uniform_(core, a=math.sqrt(5))
    #         else:  # 输入核心
    #             # 对应V矩阵的部分，使用正交初始化
    #             with torch.no_grad():
    #                 if core.size(-1) == 1:  # 最后一个核心
    #                     nn.init.orthogonal_(core.squeeze(-1))
    #                 else:
    #                     nn.init.kaiming_uniform_(core, a=math.sqrt(5))
    #
    #     if self.bias is not None:
    #         fan_in = self.in_features
    #         bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
    #         nn.init.uniform_(self.bias, -bound, bound)
    def reset_parameters(self):
        """初始化TT核心和偏置"""
        for i, core in enumerate(self.tt_cores):
            if i < self.q:  # 输出核心
                with torch.no_grad():
                    if core.size(0) == 1:  # 第一个核心
                        nn.init.kaiming_uniform_(core.squeeze(0),a=math.sqrt(0))
                    else:
                        nn.init.kaiming_uniform_(core, a=math.sqrt(0))
            else:  # 输入核心
                with torch.no_grad():
                    if core.size(-1) == 1:  # 最后一个核心
                        nn.init.kaiming_uniform_(core.squeeze(-1),a=math.sqrt(0))
                    else:
                        nn.init.kaiming_uniform_(core, a=math.sqrt(0))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.tt_cores[0].size(2))
            nn.init.uniform_(self.bias, -bound, bound)

    def reconstruct_full_weight(self):
        """
        精确模拟TT-SVD分解逆过程的重建函数。
        按照分解时的相反顺序重建完整权重。
        """
        cores = self.tt_cores
        k = len(cores)
        dims =self.out_dims+self.in_dims

        # 步骤1: 处理最后一个core (形状: r_{k-1}, d_k, 1)
        # 在TT-SVD中，最后一个core是 S_{k-1} * Vh_{k-1} 的reshape
        # print(f"最后一个核心维度为{cores[-1].shape}")
        # 重建时，我们从这里开始
        current_matrix = cores[-1].squeeze(-1)  # 形状: (r_{k-1}, d_k)
        # print(f"最后一个核心reshape后的温度为{current_matrix.shape}")

        # 步骤2: 从后向前处理每个core
        # 在TT-SVD中，每个中间步骤是: C = diag(S_i) @ Vh_i
        # 然后被reshape并与下一个core合并
        for i in range(k - 2, -1, -1):
            core_i = cores[i]  # 形状: (r_{i-1}, d_i, r_i)
            r_prev, d_i, r_i = core_i.shape

            # 在TT-SVD正向过程中，这一步是:
            # 1. 将当前矩阵reshape为 (r_i * d_{i+1}, 剩余维度)
            # 2. 进行SVD得到 U_i, S_i, Vh_i
            # 3. U_i reshape为 (r_prev, d_i, r_i) 成为core_i
            # 4. C = diag(S_i) @ Vh_i 成为下一轮迭代的矩阵

            # 逆向过程:
            # 1. core_i是U_i，形状为(r_prev, d_i, r_i)
            # 2. 我们需要将它reshape为矩阵并乘以当前矩阵

            # 将core_i reshape为矩阵: (r_prev * d_i, r_i)
            U_matrix = core_i.reshape(-1, r_i)  # 形状: (r_prev * d_i, r_i)
            # print(f"当前要合并的核心维度为{core_i.shape},该核心reshape后的维度为{U_matrix.shape},之前合并的核心维度为{current_matrix.shape}")
            # 当前矩阵是上一轮的 diag(S_i) @ Vh_i
            # 我们需要计算: U_matrix @ current_matrix
            # 这相当于重建 SVD 前的矩阵: U @ (diag(S) @ Vh) = U @ diag(S) @ Vh
            current_matrix = torch.matmul(U_matrix, current_matrix)  # 形状: (r_prev * d_i, 剩余维度)
            # 重塑以便下一步迭代
            if i > 0:  # 如果不是第一个核心
                current_matrix = current_matrix.reshape(r_prev, -1) # 形状: (r_prev, d_i * 剩余维度)


        # 4. 重塑为完整形状
        dims = self.out_dims + self.in_dims
        full_tensor = current_matrix.reshape(*dims)

        # 转换为权重矩阵格式
        in_size = math.prod(self.in_dims)
        out_size = math.prod(self.out_dims)
        weight_matrix = full_tensor.reshape(out_size,in_size)

        return weight_matrix.contiguous()


    # 这个代码有问题，二核心的时候没问题，多核心的时候重构后的前向传播行为和分解时候不同(同构实验没问题，异构实验有问题)
    # def forward(self, x):
    #     """
    #     支持输入:
    #     - (B, D)
    #     - (B, N, D)  ← ViT
    #
    #     仅对最后一个维度做 TT 线性变换
    #     """
    #     original_shape = x.shape
    #
    #     # === 统一 reshape 成 (B*, D) ===
    #     if x.dim() == 3:
    #         B, N, D = x.shape
    #         x = x.reshape(B * N, D)
    #     elif x.dim() == 2:
    #         B, D = x.shape
    #     else:
    #         raise ValueError(f"不支持的输入维度: {x.shape}")
    #
    #     batch_size = x.size(0)
    #
    #     # === TT forward（和你原来一样） ===
    #     x_reshaped = x.view(batch_size, *self.in_dims)
    #
    #     state = x_reshaped.unsqueeze(-1)  # (B*, i1,...,ip,1)
    #
    #     # === 输入核心收缩 ===
    #     for i in range(self.k - 1, self.q - 1, -1):
    #         core = self.tt_cores[i]  # (r_prev, d_i, r_next)
    #
    #         if state.size(-2) != core.size(1):
    #             raise ValueError(
    #                 f"维度不匹配: state[...,-2]={state.size(-2)}, core d_i={core.size(1)}"
    #             )
    #
    #         state = torch.einsum('...ir, kir -> ...k', state, core)
    #
    #     # === 输出核心扩展 ===
    #     for i in range(self.q - 1, -1, -1):
    #         core = self.tt_cores[i]  # (r_out, d_o, r_in)
    #
    #         if state.size(-1) != core.size(2):
    #             raise ValueError(
    #                 f"秩不匹配: state[...,-1]={state.size(-1)}, core r_in={core.size(2)}"
    #             )
    #
    #         state = torch.einsum('...r, sdr -> ...ds', state, core)
    #
    #     # === reshape 输出 ===
    #     if state.size(-1) == 1:
    #         state = state.squeeze(-1)
    #
    #     state = state.reshape(batch_size, -1)
    #
    #     # === bias ===
    #     if self.bias_flag and self.bias is not None:
    #         state = state + self.bias
    #
    #     # === 恢复 ViT 形状 ===
    #     if len(original_shape) == 3:
    #         state = state.view(B, N, -1)
    #
    #     return state

    def forward(self, x):
        """
        支持输入:
        - (B, D)
        - (B, N, D)  ← ViT
        仅对最后一个维度做 TT 线性变换
        """
        original_shape = x.shape
        # === 统一 reshape 成 (B*, D) ===
        if x.dim() == 3:
            B, N, D = x.shape
            x = x.reshape(B * N, D)
        elif x.dim() == 2:
            B, D = x.shape
        else:
            raise ValueError(f"不支持的输入维度: {x.shape}")
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, *self.in_dims)
        state = x_reshaped.unsqueeze(-1)  # (B, i1, ..., ip, 1)

        # 收缩输入核心（反向）
        for i in range(self.k - 1, self.q - 1, -1):
            core = self.tt_cores[i]
            r_prev, d_i, r_next = core.shape
            if state.size(-2) != d_i:
                raise ValueError(...)
            state = torch.einsum('...ir, kir -> ...k', state, core)

        # 扩展输出核心（反向）
        for i in range(self.q - 1, -1, -1):
            core = self.tt_cores[i]
            r_out, d_o, r_in = core.shape
            if state.size(-1) != r_in:
                raise ValueError(...)
            state = torch.einsum('...r,sdr->...ds', state, core)

        # 此时 state 形状: (B, out_{q-1}, out_{q-2}, ..., out_0, rank)
        # 反转输出维度顺序，使其与 out_dims 一致: (B, out_0, out_1, ..., out_{q-1}, rank)
        if self.q > 1:
            # 维度索引说明：
            # 0: batch
            # 1..q: 输出维度（当前顺序是反向的）
            # -1: 秩维度（通常为1）
            perm = [0] + list(range(state.dim() - 2, 0, -1)) + [state.dim() - 1]
            state = state.permute(*perm)

        # 移除秩维度（如果为1）
        if state.size(-1) == 1:
            state = state.squeeze(-1)

        # 展平输出
        state = state.reshape(batch_size, -1)

        if self.bias_flag and self.bias is not None:
            state = state + self.bias
        # === 恢复 ViT 形状 ===
        if len(original_shape) == 3:
            state = state.view(B, N, -1)
        return state

    def frobenius_loss(self):
        """计算权重矩阵的F-范数平方"""
        # 多核心情况：重建完整矩阵计算(reshap成多阶进行正则化效果一样)
        W = self.reconstruct_full_weight()
        return torch.sum(W ** 2)
    # def frobenius_loss(self):
    #     """计算权重矩阵的F-范数平方"""
    #     # 多核心情况：重建完整矩阵计算(reshap成多阶进行正则化效果一样)
    #     W = self.reconstruct_full_weight()
    #     dims = self.out_dims + self.in_dims
    #     full_tensor = W.reshape(*dims)
    #     return torch.sum(full_tensor ** 2)
    # def frobenius_loss(self):
    #     """计算所有权重核心的二范数平方之和"""
    #     total_loss = 0.0

    #     # 遍历所有TT核心（三阶张量）
    #     for i, core in enumerate(self.tt_cores):
    #         # 计算每个核心的二范数平方
    #         core_norm_sq = torch.sum(core ** 2)
    #         total_loss += core_norm_sq

    #     return total_loss

    def __repr__(self):
        return (f"TTLinear({self.in_features}, {self.out_features}, "
                f"in_dims={self.in_dims}, out_dims={self.out_dims}, "
                f"cores={self.k}, bias={self.bias_flag})")


# ---------- 分解与恢复函数 ----------
def Decom_TTLinear(linear_model, in_dims=None, out_dims=None, tt_rank_rate=None):
    """
    将普通 nn.Linear 分解为 TTLinear。
    支持用户传入 in_dims/out_dims 或使用 Factorize / FactorizeN。
    - linear_model: nn.Linear（weight shape: out_features x in_features）
    - rank_rate: 可选，若 tt_ranks 未给出则使用此值推断
    """
    device = linear_model.weight.device
    dtype = linear_model.weight.dtype

    in_features = linear_model.in_features
    out_features = linear_model.out_features
    bias = linear_model.bias is not None
    # # # 二核心
    # if in_dims is None:
    #     in_dims = [in_features]
    # if out_dims is None:
    #     out_dims = [out_features]

    # # 默认分解：输入分为两个因子，输出分为1个因子（和原版兼容） 四核
    if in_dims is None:
        in_dims = Factorize(in_features)
    if out_dims is None:
        out_dims = Factorize(out_features)
    
    # 三核心
    # if in_dims is None and out_dims is None:
    #     if in_features > out_features:
    #         in_dims = Factorize(in_features)
    #         out_dims = [out_features]
    #     else:
    #         in_dims = [in_features]
    #         out_dims = Factorize(out_features)
    print(f"输入核心张量为{in_dims},输出核心张量为{out_dims}")

    assert math.prod(in_dims) == in_features, "in_dims 乘积必须等于 in_features"
    assert math.prod(out_dims) == out_features, "out_dims 乘积必须等于 out_features"

    # 如果用户给了 rank_rate 而没给 tt_ranks，我们将在 TTLinear 中推断
    tt_linear = TTLinear(in_features, out_features,
                         in_dims=in_dims, out_dims=out_dims,
                         tt_rank_rate=tt_rank_rate,
                         bias=bias, device=device, dtype=dtype)
    tt_linear = tt_linear.to(device=device, dtype=dtype)

    # 取得原始权重并 reshape 为 (d1, d2, ..., dk)
    W_original = linear_model.weight.data.clone()  # (out_features, in_features)
    all_dims = out_dims+in_dims
    W_reshaped = W_original.reshape(*all_dims).to(device=device, dtype=dtype)

    # 如果 tt_ranks 未显式给出，这里我们可以用默认推断出的 tt_linear.tt_ranks
    cores = tt_svd(W_reshaped, tt_rank_rate, device=device, dtype=dtype)

    # 赋值
    for i, core in enumerate(cores):
        tt_linear.tt_cores[i].data.copy_(core)

    # 拷贝 bias
    if bias:
        tt_linear.bias.data.copy_(linear_model.bias.data)

    return tt_linear


def Recover_TTLinear(tt_linear):
    """将 TTLinear 恢复为普通 Linear"""
    in_features = tt_linear.in_features
    out_features = tt_linear.out_features
    bias = tt_linear.bias is not None

    W_full = tt_linear.reconstruct_full_weight()  # (out_features, in_features)
    linear_layer = nn.Linear(in_features, out_features, bias=bias).to(W_full.device, dtype=W_full.dtype)
    with torch.no_grad():
        linear_layer.weight.data.copy_(W_full)
        if bias:
            linear_layer.bias.data.copy_(tt_linear.bias.data)
    return linear_layer


# 修改Hyper_CNN类以支持TT分解
class Hyper_CNN_TT(nn.Module):
    def __init__(self, in_features=3, num_classes=10, n_kernels=16,
                 ratio_LR=0.7):
        super(Hyper_CNN_TT, self).__init__()
        self.ratio_LR = ratio_LR

        # 卷积层和池化层
        self.conv1 = nn.Conv2d(in_features, n_kernels, 5)
        if ratio_LR >= 1.0:
            self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        else:
            self.conv2 = FactorizedConv(in_channels=n_kernels, out_channels=2 * n_kernels,
                                        padding=0, rank_rate=ratio_LR, kernel_size=5, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # 计算全连接层输入维度
        self.fc_input_dim = 2 * n_kernels * 5 * 5

        # 全连接层
        if ratio_LR >= 1.0:
            self.fc1 = nn.Linear(self.fc_input_dim, 2000)
            self.fc2 = nn.Linear(2000, 500)
        else:
            self.fc1 = TTLinear(self.fc_input_dim, 2000, tt_rank_rate=ratio_LR)
            self.fc2 = TTLinear(2000, 500, tt_rank_rate=ratio_LR)

        # 激活函数和输出层
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(500, num_classes)

        # 构建共享基础部分
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
            self.relu
        )

        # 个性化头部
        self.head = self.fc3

    def TT_recover_larger_model(self):
        """将低秩/TT层恢复为完整秩"""
        if self.ratio_LR < 1:
            # 恢复TT层
            if isinstance(self.conv2, FactorizedConv):
                self.conv2 = Recover_COV(self.conv2)
            self.fc1 = Recover_TTLinear(self.fc1)
            self.fc2 = Recover_TTLinear(self.fc2)
            # 更新base索引
            self._rebuild_base()
            print("恢复低秩/TT模型为完整模型")
        else:
            return

    def TT_decom_larger_model(self, rank_rate):
        """将完整秩层分解为低秩或TT"""
        if rank_rate < 1.0:
            if isinstance(self.conv2, nn.Conv2d):
                self.conv2 = Decom_TT_COV(self.conv2, rank_rate)
            if isinstance(self.fc1, nn.Linear):
                self.fc1 = Decom_TTLinear(self.fc1, tt_rank_rate=rank_rate)
            if isinstance(self.fc2, nn.Linear):
                self.fc2 = Decom_TTLinear(self.fc2, tt_rank_rate=rank_rate)
            self._rebuild_base()
            print(f"将完整模型进行TT分解分解比例为{rank_rate}")
        else:
            # 不需要分解
            return

    def _rebuild_base(self):
        """重构基础网络部分"""
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
            self.relu
        )

    def frobenius_decay(self):
        """计算Frobenius衰减"""
        total_loss = torch.tensor(0.0, device=self.conv1.weight.device)

        if self.ratio_LR < 1:
            total_loss += self.fc1.frobenius_loss() + self.fc2.frobenius_loss() + self.conv2.frobenius_loss()

        return total_loss

    def forward(self, x):
        features = self.base(x)
        output = self.head(features)
        return output


class Hyper_CNN_tiny_TT(nn.Module):
    def __init__(self, in_features=3, num_classes=10, n_kernels=16,
                 ratio_LR=0.7):
        super(Hyper_CNN_tiny_TT, self).__init__()
        self.ratio_LR = ratio_LR

        # 卷积层和池化层
        self.conv1 = nn.Conv2d(in_features, n_kernels, 5)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # 计算全连接层输入维度
        self.fc_input_dim = 5408

        # 全连接层
        if ratio_LR >= 1.0:
            self.fc1 = nn.Linear(self.fc_input_dim, 2000)
            self.fc2 = nn.Linear(2000, 500)
        else:
            self.fc1 = TTLinear(self.fc_input_dim, 2000, tt_rank_rate=ratio_LR)
            self.fc2 = TTLinear(2000, 500, tt_rank_rate=ratio_LR)

        # 激活函数和输出层
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(500, num_classes)

        # 构建共享基础部分
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
            self.relu
        )

        # 个性化头部
        self.head = self.fc3

    def TT_recover_larger_model(self):
        """将低秩/TT层恢复为完整秩"""
        if self.ratio_LR < 1:
            self.fc1 = Recover_TTLinear(self.fc1)
            self.fc2 = Recover_TTLinear(self.fc2)
            # 更新base索引
            self._rebuild_base()
            print("恢复低秩/TT模型为完整模型")
        else:
            return

    def TT_decom_larger_model(self, rank_rate):
        """将完整秩层分解为低秩或TT"""
        if rank_rate < 1.0:
            if isinstance(self.fc1, nn.Linear):
                self.fc1 = Decom_TTLinear(self.fc1, tt_rank_rate=rank_rate)
            if isinstance(self.fc2, nn.Linear):
                self.fc2 = Decom_TTLinear(self.fc2, tt_rank_rate=rank_rate)
            self._rebuild_base()
            print(f"将完整模型进行TT分解分解比例为{rank_rate}")
        else:
            # 不需要分解
            return

    def _rebuild_base(self):
        """重构基础网络部分"""
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
            self.relu
        )

    def frobenius_decay(self):
        """计算Frobenius衰减"""
        total_loss = torch.tensor(0.0, device=self.conv1.weight.device)

        if self.ratio_LR < 1.0:
            total_loss += self.fc1.frobenius_loss() + self.fc2.frobenius_loss()

        return total_loss

    def forward(self, x):
        features = self.base(x)
        output = self.head(features)
        return output
class CNN_1_512(BaseHeadCNN):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        # 基础部分
        base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 2000),
            nn.ReLU(),
            nn.Linear(2000, 512),
        )
        
        # 分类器头
        head = nn.Sequential(
            nn.ReLU(),  
            nn.Linear(512, out_dim),)
        
        super().__init__(base, head)

# 原始CNN_2重构
class CNN_2_512(BaseHeadCNN):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(n_kernels * 5 * 5, 2000),
            nn.ReLU(),
            nn.Linear(2000, 512),
        )
        
        head =  nn.Sequential(
            nn.ReLU(),  
            nn.Linear(512, out_dim)
        )
        super().__init__(base, head)

# 原始CNN_3重构
class CNN_3_512(BaseHeadCNN):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
        )
        
        head = nn.Sequential(
            nn.ReLU(),  
            nn.Linear(512, out_dim)
        )
        super().__init__(base, head)

# 原始CNN_4重构
class CNN_4_512(BaseHeadCNN):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 800),
            nn.ReLU(),
            nn.Linear(800, 512),
        )
        
        head = nn.Sequential(
            nn.ReLU(),  
            nn.Linear(512, out_dim)
        )
        super().__init__(base, head)

# 原始CNN_5重构
class CNN_5_512(BaseHeadCNN):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 500),
            nn.ReLU(),
            nn.Linear(500, 512),
        )
        
        head = nn.Sequential(
            nn.ReLU(),  
            nn.Linear(512, out_dim)
        )
        super().__init__(base, head)



class Hyper_CNN_512(nn.Module):
    def __init__(self, in_features=3, num_classes=10, n_kernels=16, ratio_LR=0.7):
        super(Hyper_CNN_512, self).__init__()
        self.ratio_LR = ratio_LR

        # 卷积层和池化层
        self.conv1 = nn.Conv2d(in_features, n_kernels, 5)
        if ratio_LR >= 1.0:
            self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        else:
            self.conv2 = FactorizedConv(in_channels=n_kernels, out_channels=2* n_kernels, padding=0,rank_rate=ratio_LR,kernel_size=5,bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # 计算全连接层输入维度
        self.fc_input_dim = 2 * n_kernels * 5 * 5
        # #
        # print(f"全连接层的输入维度为：",self.fc_input_dim)
        # 全连接层（可能低秩分解）
        if ratio_LR >= 1.0:
            self.fc1 = nn.Linear(self.fc_input_dim, 2000)
            self.fc2 = nn.Linear(2000, 512)
        else:
            self.fc1 = FactorizedLinear(in_features=self.fc_input_dim, out_features=2000, rank_rate=ratio_LR, bias=True)
            self.fc2 = FactorizedLinear(2000,512, rank_rate=ratio_LR, bias=True)

        # 激活函数和输出层
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(512, num_classes)

        # 构建共享基础部分
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
        )

        # 个性化头部
        self.head = nn.Sequential(
            self.relu,  
            self.fc3
        )

    def recover_larger_model(self):
        """将低秩层恢复为完整秩"""
        if self.ratio_LR >= 1.0:
            return
        self.conv2 = Recover_COV(self.conv2)
        # 恢复两个全连接层
        self.fc1 = Recover_LINEAR(self.fc1)
        self.fc2 = Recover_LINEAR(self.fc2)
        # 更新base索引
        self._rebuild_base()
        print("(卷积)恢复低秩模型为完整模型，fc1和fc2已恢复")

    def decom_larger_model(self, rank_rate):
        """将完整秩层分解为低秩"""
        if rank_rate >= 1.0:
            return

        if isinstance(self.conv2,nn.Conv2d):
            self.conv2 = Decom_COV(self.conv2,rank_rate)
        if isinstance(self.fc1, nn.Linear):
            self.fc1 = Decom_LINEAR(self.fc1, rank_rate)
        if isinstance(self.fc2, nn.Linear):
            self.fc2 = Decom_LINEAR(self.fc2, rank_rate)

        self._rebuild_base()
        print(f"将完整模型分解(卷积也分解)为低秩模型(rank_rate={rank_rate})")

    def _rebuild_base(self):
        """重构基础网络部分"""
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
        )

    def frobenius_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.frobenius_loss()+self.fc2.frobenius_loss() + self.conv2.frobenius_loss()

    def kronecker_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.kronecker_loss()+self.fc2.kronecker_loss() + self.conv2.kronecker_loss()

    def L2_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.L2_loss() +self.fc2.L2_loss()+ self.conv2.L2_loss()

    def forward(self, x):
        features = self.base(x)  # 提取特征
        output = self.head(features)  # 分类输出
        return output


# ---------------------FedAFM-----------------------------
class CNN_1_hetero_AFM_512(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_1_hetero_AFM_512, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 2000),
            nn.ReLU(),
            nn.Linear(2000, 512),
        )

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x, homo_rep, alpha):
        feature = self.base(x)
        mix_feature = feature*alpha.to(homo_rep.device) + homo_rep
        output = self.head(mix_feature)
        return output,mix_feature

class CNN_2_hetero_AFM_512(nn.Module): # change filters of convs
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_2_hetero_AFM_512, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(n_kernels * 5 * 5, 2000),
            nn.ReLU(),
            nn.Linear(2000, 512),
        )
        
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x, homo_rep, alpha):
        feature = self.base(x)
        mix_feature = feature*alpha.to(homo_rep.device) + homo_rep
        output = self.head(mix_feature)
        return output,mix_feature

class CNN_3_hetero_AFM_512(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_3_hetero_AFM_512, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
        )
        
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x, homo_rep, alpha):
        feature = self.base(x)
        mix_feature = feature*alpha.to(homo_rep.device) + homo_rep
        output = self.head(mix_feature)
        return output,mix_feature


class CNN_4_hetero_AFM_512(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_4_hetero_AFM_512, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 800),
            nn.ReLU(),
            nn.Linear(800, 512),
        )
        
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    def forward(self, x, homo_rep, alpha):
        feature = self.base(x)
        mix_feature = feature*alpha.to(homo_rep.device) + homo_rep
        output = self.head(mix_feature)
        return output,mix_feature

class CNN_5_hetero_AFM_512(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_5_hetero_AFM_512, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 500),
            nn.ReLU(),
            nn.Linear(500, 512),
        )
        
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x, homo_rep, alpha):
        feature = self.base(x)
        mix_feature = feature*alpha.to(homo_rep.device) + homo_rep
        output = self.head(mix_feature)
        return output,mix_feature

class CNN_5_homo_AFM_512(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_5_homo_AFM_512, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * n_kernels * 5 * 5, 500),
            nn.ReLU(),
            nn.Linear(500, 512),
        )
        
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        feature = self.base(x)
        output = self.head(feature)
        return output, feature