#!/usr/bin/env python
import torch
import argparse
import os
import time
import warnings
import numpy as np
import logging
from datetime import datetime
from flcore.servers.serverspu import FedSPU
from flcore.servers.serverlocal import Local
from flcore.servers.serverproto import FedProto
from flcore.servers.servergen import FedGen
from flcore.servers.serverfd import FD
from flcore.servers.serverlg import LG_FedAvg
from flcore.servers.serverfml import FML
from flcore.servers.serverkd import FedKD
from flcore.servers.servergh import FedGH
from flcore.servers.servertgp import FedTGP
from flcore.servers.serverktl_stylegan_xl import FedKTL as FedKTL_stylegan_xl
from flcore.servers.serverktl_stylegan_3 import FedKTL as FedKTL_stylegan_3
from flcore.servers.serverktl_stable_diffusion import FedKTL as FedKTL_stable_diffusion
from flcore.servers.servermrl import FedMRL
from flcore.servers.serverwz import FedWZ
from flcore.servers.serverHAS import FedHAS
from flcore.servers.serveradra import ADRALPFL
from flcore.servers.serverafm import PFedAFM
from flcore.servers.serverARA2 import FedARA2
from flcore.servers.serverDAR import FedDAR
from flcore.servers.serversce import Fedsce
from flcore.servers.serverCLIP import FedCLIP
from flcore.servers.serverPer import FedPer
from flcore.servers.serveravg import Fedavg
from utils.result_utils import average_data
from utils.mem_utils import MemReporter
#日志文件
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0) 

def run(args):

    time_list = []
    reporter = MemReporter()

    for i in range(args.prev, args.times): # 可能跑多次取平均值
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.models  设置异构模型架构
        if args.model_family == "HtFE-img-2":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)', # for 32x32 img
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
            ]

        elif args.model_family == "HtFE-img-3":
            args.models = [
                'resnet10(num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
            ]

        elif args.model_family == "HtFE-img-4":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)', # for 32x32 img
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)'
            ]

        elif args.model_family == "HtFE-img-5":
            args.models = [
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
            ]

        elif args.model_family == "HtFE-img-8":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)', # for 32x32 img
                # 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816)', # for 64x64 img
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
            ]

        elif args.model_family == "HtFE-img-9":
            args.models = [
                'resnet4(num_classes=args.num_classes)', 
                'resnet6(num_classes=args.num_classes)', 
                'resnet8(num_classes=args.num_classes)', 
                'resnet10(num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)', 
            ]
        #这个是什么异构设置  异构特征提取器和异构特征分类器    两部分都异构
        elif args.model_family == "HtFE-img-8-HtC-img-4":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)', # for 32x32 img
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
            ]
            args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)'# for 32x32 img
            args.heads = [
                'Head(hidden_dims=[512], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 512], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 256], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 128], num_classes=args.num_classes)', 
            ]
        #同构特征提取器  异构分类器
        elif args.model_family == "Res34-HtC-img-4":
            args.models = [
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
            ]
            args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)'# for 32x32 img
            args.heads = [
                'Head(hidden_dims=[512], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 512], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 256], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 128], num_classes=args.num_classes)', 
            ]

        elif args.model_family == "HtM-img-10":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)', # for 32x32 img
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.vit_b_16(image_size=32, num_classes=args.num_classes)', 
                'torchvision.models.vit_b_32(image_size=32, num_classes=args.num_classes)'
            ]
        #文本异构设置
        elif args.model_family == "HtFE-txt-2":
            args.models = [
                'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                'TextLogisticRegression(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)'
            ]

        elif args.model_family == "HtFE-txt-4":
            args.models = [
                'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                'TextLogisticRegression(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                'LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                'BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, output_size=args.num_classes, num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, embedding_length=args.feature_dim)'
            ]

        elif args.model_family == "HtFE-txt-5-1":
            args.models = [
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)',
            ]

        elif args.model_family == "HtFE-txt-5-2":
            args.models = [
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
            ]

        elif args.model_family == "HtFE-txt-5-3":
            args.models = [
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)',
            ]
        
        elif args.model_family == "HtFE-txt-6":
            args.models = [
                'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)', 
                'LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)', 
                'BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, output_size=args.num_classes, num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, embedding_length=args.feature_dim)', 
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
                'TextLogisticRegression(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                'GRUNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)'
            ]
        #
        elif args.model_family == "MLPs":
            args.models = [
                'AmazonMLP(feature_dim=[])', 
                'AmazonMLP(feature_dim=[500])', 
                'AmazonMLP(feature_dim=[1000, 500])', 
                'AmazonMLP(feature_dim=[1000, 500, 200])', 
            ]

        elif args.model_family == "MLP_1layer":
            args.models = [
                'AmazonMLP(feature_dim=[200])', 
                'AmazonMLP(feature_dim=[500])', 
            ]

        elif args.model_family == "MLP_layers":
            args.models = [
                'AmazonMLP(feature_dim=[500])', 
                'AmazonMLP(feature_dim=[1000, 500])', 
                'AmazonMLP(feature_dim=[1000, 500, 200])', 
            ]

        elif args.model_family == "HtFE-sen-2":
            args.models = [
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
            ]

        elif args.model_family == "HtFE-sen-3":
            args.models = [
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=3)',
            ]

        elif args.model_family == "HtFE-sen-5":
            args.models = [
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=3)',
                'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=1)',
                'HARCNN3(9, dim_hidden=3328, num_classes=args.num_classes, stride=1)',
            ]

        elif args.model_family == "HtFE-sen-8":
            args.models = [
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=3)',
                'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=1)',
                'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=2)',
                'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=3)',
                'HARCNN3(9, dim_hidden=3328, num_classes=args.num_classes, stride=1)',
                'HARCNN3(9, dim_hidden=3328, num_classes=args.num_classes, stride=2)',
            ]
        elif args.model_family == "Decom_CNN-5":
            args.models = [
                'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=1.0)',
                'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.5)',
                'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.35)',
                'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.25)',
                'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.15)',

            ]
            args.global_model = 'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.15)'
        elif args.model_family == "CNN-5":
            args.models = [
                'CNN_1(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_2(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_3(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_4(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_5(in_channels=3, n_kernels=16, out_dim=args.num_classes)',

            ]
            args.global_model = 'CNN_5(in_channels=3, n_kernels=16, out_dim=args.num_classes)'
        elif args.model_family == "CNN-5-tiny":
            args.models = [
                'CNN_1_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_2_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_3_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_4_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_5_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
            ]
            args.global_model = 'CNN_5_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)'
        elif args.model_family == "Decom_CNN-5-tiny":
            args.models = [
                'Hyper_CNN_tiny(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=1.0)',
                'Hyper_CNN_tiny(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.4)',
                'Hyper_CNN_tiny(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.35)',
                'Hyper_CNN_tiny(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.29)',
                'Hyper_CNN_tiny(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.18)',

            ]
            args.global_model = 'Hyper_CNN_tiny(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.18)'
        #跑FedAFM使用的模型
        elif args.model_family == "CNN-5-AFM":
            args.models = [
                'CNN_1_hetero_AFM(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_2_hetero_AFM(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_3_hetero_AFM(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_4_hetero_AFM(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_5_hetero_AFM(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
            ]
            args.global_model = 'CNN_5_homo_AFM(in_channels=3, n_kernels=16, out_dim=args.num_classes)'
        elif args.model_family == "CNN-5-AFM-tiny":
            args.models = [
                'CNN_1_hetero_AFM_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_2_hetero_AFM_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_3_hetero_AFM_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_4_hetero_AFM_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_5_hetero_AFM_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
            ]
            args.global_model = 'CNN_5_homo_AFM_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)'
        elif args.model_family == "homo_FedavgCNN":
            args.models = [
                'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=1.0)'  
            ]
            args.global_model = 'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=1.0)'
        elif args.model_family == "FedavgCNN":
            args.models = [
                'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=0.5)'  
            ]
            args.global_model = 'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=0.5)'
        elif args.model_family == "FedavgCNN_AFM":
            args.models = [
                'FedAvgCNN_Hetero_AFM(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=1.0)'  
            ]
            args.global_model = 'FedAvgCNN_Homo_AFM(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=1.0)'
        elif args.model_family == "SPU_CNN1":
            args.models = [
                'CNN_1(in_channels=3, n_kernels=16, out_dim=args.num_classes)'  
            ]
            args.global_model = 'CNN_1(in_channels=3, n_kernels=16, out_dim=args.num_classes)'   
        elif args.model_family == "homo_FedavgCNN-tiny":
            args.models = [
                'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816, ratio_LR=1.0)'  
            ]
            args.global_model = 'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816, ratio_LR=1.0)'
        elif args.model_family == "Decom_FedavgCNN-tiny":
            args.models = [
                'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816, ratio_LR=0.5)'  
            ]
            args.global_model = 'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816, ratio_LR=0.5)'
        elif args.model_family == "FedavgCNN_AFM-tiny":
            args.models = [
                'FedAvgCNN_Hetero_AFM(in_features=3, num_classes=args.num_classes, dim=10816, ratio_LR=1.0)'  
            ]
            args.global_model = 'FedAvgCNN_Homo_AFM(in_features=3, num_classes=args.num_classes, dim=10816, ratio_LR=1.0)'
        elif args.model_family == "SPU_CNN1-tiny":
            args.models = [
                'CNN_1_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)'  
            ]
            args.global_model = 'CNN_1_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)'  
        elif args.model_family == "TT_CNN":
            args.models = [
                'Hyper_CNN_TT(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.5)',
            ]
            args.global_model = 'Hyper_CNN_TT(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.15)'
        elif args.model_family == "TT_CNN-tiny":
            args.models = [
                'Hyper_CNN_tiny_TT(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.4)',

            ]
            args.global_model = 'Hyper_CNN_tiny_TT(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.18)'
        # 不同系统异构程度实验
        elif args.model_family == "Level1-Decom_CNN":
            args.models = [
                'CNN_1(in_channels=3, n_kernels=16, out_dim=args.num_classes)'  
            ]
            args.global_model = 'CNN_1(in_channels=3, n_kernels=16, out_dim=args.num_classes)'
        elif args.model_family == "Level2-Decom_CNN":
            args.models = [
                'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=1.0)'  
            ]
            args.global_model = 'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=1.0)'
        elif args.model_family == "Level3-Decom_CNN":
            args.models = [
                'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=0.35)'  
            ]
            args.global_model = 'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=0.5)'
        elif args.model_family == "Level4-Decom_CNN":
            args.models = [
                'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=0.25)'  
            ]
            args.global_model = 'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=0.5)'
        elif args.model_family == "Level5-Decom_CNN":
            args.models = [
                'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=0.15)'  
            ]
            args.global_model = 'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=0.5)'
        elif args.model_family == "Level6-Decom_CNN":
            args.models = [
                'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=0.10)'  
            ]
            args.global_model = 'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=0.5)'
        elif args.model_family == "Level7-Decom_CNN":
            args.models = [
                'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=0.05)'  
            ]
            args.global_model = 'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=0.5)'
        elif args.model_family == "Level8-Decom_CNN":
            args.models = [
                'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=0.03)'  
            ]
            args.global_model = 'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=0.5)'
        elif args.model_family == "Level9-Decom_CNN":
            args.models = [
                'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=0.01)'  
            ]
            args.global_model = 'Decom_FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=0.5)'
        elif args.model_family == "Level10-Decom_CNN":
            args.models = [
                'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.0625)',
            ]
            args.global_model = 'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.3)'
        elif args.model_family == "Level11-Decom_CNN":
            args.models = [
                'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.0625)',
                'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.05)',
            ]
            args.global_model = 'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.3)'
        elif args.model_family == "Level12-Decom_CNN":
            args.models = [
                'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.05)',
            ]
            args.global_model = 'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.3)'
        elif args.model_family == "Levelf-Decom_CNN":
            args.models = [
                'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.5)',
            ]
            args.global_model = 'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.15)'
        elif args.model_family == "Level13-Decom_CNN":
            args.models = [
                'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.05)',
            ]
            args.global_model = 'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.05)'
        elif args.model_family == "Level14-Decom_CNN":
            args.models = [
                'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.03)',
            ]
            args.global_model = 'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.03)'
        elif args.model_family == "Level15-Decom_CNN":
            args.models = [
                'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.01)',
            ]
            args.global_model = 'Hyper_CNN(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.01)'
        elif args.model_family == "resnet":
            args.models = [
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
            ]
            args.global_model ='torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)'   
        elif args.model_family == "low_rank_resnet":
            args.models = [
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=group_norm,has_norm = True,bn_block_num = 4, ratio_LR = 1.0)',
            ]
            args.global_model ='low_rank_resnet18_cifar(features=[64, 128, 256, 512],num_classes=args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation = None,norm_layer=group_norm, has_norm = True,bn_block_num = 4, ratio_LR = 1.0)'    
        elif args.model_family == "low_rank_resnet_mutil":
            args.models = [
                'low_rank_resnet18_cifar_MUTIL(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=group_norm,has_norm = True,bn_block_num = 4, ratio_LR = 1.0)',
            ]
            args.global_model ='low_rank_resnet18_cifar_MUTIL(features=[64, 128, 256, 512],num_classes=args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation = None,norm_layer=group_norm, has_norm = True,bn_block_num = 4, ratio_LR = 1.0)'   
        elif args.model_family == "low_rank_resnet_512":
            args.models = [
                'low_rank_resnet8_cifar_512(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=group_norm,has_norm = True,bn_block_num = 4, ratio_LR = 1.0)',
            ]
            args.global_model ='low_rank_resnet8_cifar_512(features=[64, 128, 256, 512],num_classes=args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation = None,norm_layer=group_norm, has_norm = True,bn_block_num = 4, ratio_LR = 1.0)'    
        elif args.model_family == "low_rank_resnet_5":
            args.models = [
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = False,bn_block_num = 4, ratio_LR = 1.0)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = False,bn_block_num = 4, ratio_LR = 0.4)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = False,bn_block_num = 4, ratio_LR = 0.2)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = False,bn_block_num = 4, ratio_LR = 0.12)',   
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = False,bn_block_num = 4, ratio_LR = 0.05)',       
            ]
            args.global_model ='low_rank_resnet18_cifar(features=[64, 128, 256, 512],num_classes=args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation = None,norm_layer=None, has_norm = False,bn_block_num = 4, ratio_LR = 0.05)'                
        elif args.model_family == "low_rank_resnet_5_bn":
            args.models = [
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = True,bn_block_num = 4, ratio_LR = 1.0)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = True,bn_block_num = 4, ratio_LR = 0.4)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = True,bn_block_num = 4, ratio_LR = 0.2)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = True,bn_block_num = 4, ratio_LR = 0.12)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = True,bn_block_num = 4, ratio_LR = 0.05)',         
            ]
            args.global_model ='low_rank_resnet18_cifar(features=[64, 128, 256, 512],num_classes=args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation = None,norm_layer=None, has_norm = True,bn_block_num = 4, ratio_LR = 0.05)'     
        elif args.model_family == "low_rank_resnet_5_in":
            args.models = [
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=instance_norm,has_norm = True,bn_block_num = 4, ratio_LR = 1.0)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=instance_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.4)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=instance_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.2)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=instance_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.12)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=instance_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.05)',         
            ]
            args.global_model ='low_rank_resnet18_cifar(features=[64, 128, 256, 512],num_classes=args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation = None,norm_layer=instance_norm, has_norm = True,bn_block_num = 4, ratio_LR = 0.05)'     
        elif args.model_family == "low_rank_resnet_5_gn":
            args.models = [
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=group_norm,has_norm = True,bn_block_num = 4, ratio_LR = 1.0)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=group_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.4)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=group_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.2)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=group_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.12)',    
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=group_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.05)',                      
            ]
            args.global_model ='low_rank_resnet18_cifar(features=[64, 128, 256, 512],num_classes=args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation = None,norm_layer=group_norm, has_norm = True,bn_block_num = 4, ratio_LR = 0.05)'         
        elif args.model_family == "resnet_5":
            args.models = [
                'low_rank_resnet18_cifar_512(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = False,bn_block_num = 4, ratio_LR = 1.0)',
                'CNN_2_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'low_rank_resnet10_cifar_512(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = False,bn_block_num = 4, ratio_LR = 1.0)',
                'CNN_5_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'low_rank_resnet8_cifar_512(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = False,bn_block_num = 4, ratio_LR = 1.0)',        
            ]
            args.global_model ='low_rank_resnet8_cifar_512(features=[64, 128, 256, 512],num_classes=args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation = None,norm_layer= None, has_norm = False,bn_block_num = 4, ratio_LR = 1.0)'        
        elif args.model_family == "VIT":
            args.models = [
                'ViT(image_size=32,patch_size=4,num_classes=args.num_classes,dim=384,depth=1,heads=6,mlp_dim=1536,dim_head=64,dropout=0.3,emb_dropout=0.3,pool=\'cls\',channels=3)'
            ]
            args.global_model ='ViT(image_size=32,patch_size=4,num_classes=args.num_classes,dim=384,depth=1,heads=6,mlp_dim=1536,dim_head=64,dropout=0.3,emb_dropout=0.3,pool=\'cls\',channels=3,)'        
        elif args.model_family == "LOW_RANK_VIT":
            args.models = [
                'LOW_RANK_ViT(image_size=32,patch_size=4,num_classes=args.num_classes,dim=384,depth=6,heads=6,mlp_dim=1536,dim_head=64,dropout=0.0,emb_dropout=0.0,pool=\'mean\',channels=3,ratio_LR=1.0)'
            ]
            args.global_model ='LOW_RANK_ViT(image_size=32,patch_size=4,num_classes=args.num_classes,dim=384,depth=6,heads=6,mlp_dim=1536,dim_head=64,dropout=0.0,emb_dropout=0.0,pool=\'mean\',channels=3,ratio_LR=1.0)'        
        elif args.model_family == "Decom_LOW_RANK_VIT":
            args.models = [
                'LOW_RANK_ViT_Select(image_size=32,patch_size=4,num_classes=args.num_classes,dim=384,depth=1,heads=6,mlp_dim=1536,dim_head=64,dropout=0.3,emb_dropout=0.3,pool=\'cls\',channels=3,ratio_LR=0.5,decom_start_layer=2)'
            ]
            args.global_model ='LOW_RANK_ViT_Select(image_size=32,patch_size=4,num_classes=args.num_classes,dim=384,depth=1,heads=6,mlp_dim=1536,dim_head=64,dropout=0.3,emb_dropout=0.3,pool=\'cls\',channels=3,ratio_LR=0.15,decom_start_layer=2)'        
        elif args.model_family == "Decom_LOW_RANK_Swin":
            args.models = [
                'Low_Rank_SwinTransformer(img_size=32,patch_size=2,in_chans=3,num_classes=args.num_classes,embed_dim=64,depths=[2, 2, 2],num_heads=[2, 4, 8],window_size=4,mlp_ratio=4.0,drop_rate=0.0,attn_drop_rate=0.0,drop_path_rate=0.1,patch_norm=True,ratio_LR=1.0)'
            ]
            args.global_model ='Low_Rank_SwinTransformer(img_size=32,patch_size=2,in_chans=3,num_classes=args.num_classes,embed_dim=64,depths=[2, 2, 2],num_heads=[2, 4, 8],window_size=4,mlp_ratio=4.0,drop_rate=0.0,attn_drop_rate=0.0,drop_path_rate=0.1,patch_norm=True,ratio_LR=1.0)'        
        elif args.model_family == "Decom_CNN-5-512":
            args.models = [
                'Hyper_CNN_512(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=1.0)', # 暂时只考虑一个秩
                'Hyper_CNN_512(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.5)',
                'Hyper_CNN_512(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.35)',
                'Hyper_CNN_512(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.25)',
                'Hyper_CNN_512(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.15)',
            ]
            args.global_model = 'Hyper_CNN_512(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.15)'
        elif args.model_family == "CNN-5-512":
            args.models = [
                'CNN_1_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_2_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_3_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_4_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_5_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)', 
            ]
            args.global_model = 'CNN_5_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)'      
        elif args.model_family == "CNN-5-512-AFM":
            args.models = [
                'CNN_1_hetero_AFM_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_2_hetero_AFM_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_3_hetero_AFM_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_4_hetero_AFM_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_5_hetero_AFM_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
            ]
            args.global_model = 'CNN_5_homo_AFM_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)'        
        else:
            raise NotImplementedError
        #客户端不同的模型架构
        for model in args.models:
            print("-------------------------------------客户端使用的模型架构----------------------------------------")
            print(model)
        #全局模型架构
        if hasattr(args, 'global_model'):
            print('global_model:', args.global_model)
        #分类器设置
        if hasattr(args, 'heads'):
            for head in args.heads:
                print('head:', head)
        #在此添加算法
        # select algorithm
        if args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedProto":
            server = FedProto(args, i)

        elif args.algorithm == "FedGen":
            server = FedGen(args, i)

        elif args.algorithm == "FD":
            server = FD(args, i)

        elif args.algorithm == "LG-FedAvg":
            server = LG_FedAvg(args, i)

        elif args.algorithm == "FML":
            server = FML(args, i)

        elif args.algorithm == "FedKD":
            server = FedKD(args, i)

        elif args.algorithm == "FedGH":
            server = FedGH(args, i)

        elif args.algorithm == "FedTGP":
            server = FedTGP(args, i)
            
        elif args.algorithm == "FedKTL-stylegan-xl":
            server = FedKTL_stylegan_xl(args, i)

        elif args.algorithm == "FedKTL-stylegan-3":
            server = FedKTL_stylegan_3(args, i)

        elif args.algorithm == "FedKTL-stable-diffusion":
            server = FedKTL_stable_diffusion(args, i)

        elif args.algorithm == "FedMRL":
            server = FedMRL(args, i)
        #在此实现自己的算法
        elif args.algorithm == 'FedWZ':
            server = FedWZ(args, i)
        elif args.algorithm == 'FedHAS':
            server = FedHAS(args, i)
        elif args.algorithm == 'PFedAFM':
            server = PFedAFM(args, i)
        elif args.algorithm == 'ADRALPFL':
            server = ADRALPFL(args, i)
        elif args.algorithm == 'FedSPU':
            server = FedSPU(args, i)
        elif args.algorithm == 'FedARA2':
            server = FedARA2(args, i)
        elif args.algorithm == 'FedSCE':
            server = Fedsce(args, i)
        elif args.algorithm == 'FedDAR':
            server = FedDAR(args, i)
        elif args.algorithm == 'FedCLIP':
            server = FedCLIP(args, i)
        elif args.algorithm == 'FedPer':
            server = FedPer(args, i)
        elif args.algorithm == 'FedAVG':
            server = Fedavg(args, i)
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general 通用参数  实验目标跑的时候设置为 train
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    #指定实验使用GPU还是CPU
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    #GPU号
    parser.add_argument('-did', "--device_id", type=str, default="0")
    #实验数据集
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    #数据集类别数
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    #指定客户端模型复杂度
    parser.add_argument('-m', "--model_family", type=str, default="HtM10")
    #本地批次大小
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    #本地学习率
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    #学习率衰减
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    #衰减采纳数
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    #全局通信轮次
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    #早停设置1轮次
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    #本地训练轮次
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    #算法名称
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    #参与比例
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    #是否设置随机参与
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    #客户端个数
    parser.add_argument('-nc', "--num_clients", type=int, default=2,
                        help="Total number of clients")
    #
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    #
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    #多少个通信轮次评估一次
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    #结果文件保存位置
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='temp')
    #是否早停
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    #特征维度
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    #文本任务  字典大小
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    #句子最大长度
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    #模型存放文件夹
    parser.add_argument('-mfn', "--models_folder_name", type=str, default='',
                        help="The folder of pre-trained models")
    #从训练数据集采样部分数据？
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    # practical  模拟真实世界参数
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    parser.add_argument('-exp_name', "--exp_name", type=str, default="FedAVG")

    # FedProto  特有参数
    parser.add_argument('-lam', "--lamda", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=100)
    # FML
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    parser.add_argument('-bt', "--beta", type=float, default=1.0)
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.01)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedGH  服务器微调head的学习率
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=0.01)
    # FedTGP
    parser.add_argument('-mart', "--margin_threthold", type=float, default=100.0)
    # FedKTL
    parser.add_argument('-GPath', "--generator_path", type=str, default='stylegan/stylegan-xl-models/imagenet64.pkl')
    parser.add_argument('-prompt', "--stable_diffusion_prompt", type=str, default='a cat')
    parser.add_argument('-sbs', "--server_batch_size", type=int, default=100)
    parser.add_argument('-gbs', "--gen_batch_size", type=int, default=4,
                        help="Not related to the performance. A small value saves GPU memory.")
    parser.add_argument('-mu', "--mu", type=float, default=50.0)
    # FedMRL
    parser.add_argument('-sfd', "--sub_feature_dim", type=int, default=128)

    #FedWZ FedARA
    parser.add_argument('-align_epoch', "--align_epoch", type=int, default=1)
    parser.add_argument('-is_regular', "--is_regular", type=int, default=0)
    parser.add_argument('-regular_lamda', "--regular_lamda", type=float, default=5e-4)
    parser.add_argument('-align_lr', "--align_lr", type=float, default=0.01)
    parser.add_argument('-a_m', "--align_method", type=int, default=1)
    parser.add_argument('-klT', "--kl_Tim", type=int, default=1)
    parser.add_argument('-kl_lamda', "--kl_lamda", type=float, default=0.1)
    parser.add_argument('-mse_lamda', "--mse_lamda", type=float, default=1.0)
    #PFedAFM
    parser.add_argument('-alpha_lr', "--alpha_lr", type=float, default=0.01)
    #是否进行本地对齐
    parser.add_argument('-wo_local','--wo_local',  type=int, default=1)
    
    # FedSCE
    parser.add_argument('-rank','--rank', type=int, default=3, help="rank parameter")
    parser.add_argument('-l_i', "--layer_idx", type=int, default=30,help="layer number of subspace")
    parser.add_argument('-gap',"--gap", type=int, default=5, help="The gap between subspace update")
    parser.add_argument('-sce_lam', "--sce_lamda", type=float, default=0.1)
    
    #DAR 
    parser.add_argument('-Con_lamda','--Con_lamda',  type=float, default=1.0)
    parser.add_argument('-Rel_lamda','--Rel_lamda',  type=float, default=1.0) 
    parser.add_argument('-which_con','--which_con',  type=str, default='sim')
    parser.add_argument('-hard_negative_mining', '--hard_negative_mining', 
                    action='store_true', default=False,
                    help='是否使用硬负样本挖掘 (默认为False)') 
    parser.add_argument('-is_TT_Decom', '--is_TT_Decom', 
                    action='store_true', default=False,
                    help='是否使用TT分解 (默认为False)') 
    parser.add_argument('-topk','--topk',  type=int, default=5)
    parser.add_argument('-Con_T','--Con_T',  type=float, default=0.1) 

    # CLIP contrastive_lamda
    parser.add_argument('-Cos_lamda','--Cos_lamda',  type=float, default=1.0)
    parser.add_argument('-temperature','--temperature',  type=float, default=0.1)
    parser.add_argument('-struct_lamda','--struct_lamda',  type=float, default=0.1)
    parser.add_argument('-rel_lamda','--rel_lamda',  type=float, default=0.1)
    # === 新增：数据异构性控制参数 ===
    parser.add_argument('-niid', "--niid", type=int, default=1, help="1 for Non-IID, 0 for IID")
    parser.add_argument('-pt', "--partition", type=str, default="dir", choices=['dir', 'pat', 'exdir'], help="Partition strategy")
    parser.add_argument('-dir_alpha', "--dir_alpha", type=float, default=0.1, help="Dirichlet coefficient (alpha)")
    parser.add_argument('-cpc', "--class_per_client", type=int, default=6, help="Classes per client (for pat)")
    # === 新增：断点续训控制参数 ===
    parser.add_argument('-resume', '--resume', action='store_true', default=False, 
                        help="是否从上一次意外中断的 checkpoint 继续训练")
        
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    # 获取当前时间并格式化为 "YYYY-MM-DD HH:MM:SS"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args.exp_name = f"algo_{args.algorithm}-dataset{args.dataset}-{current_time}"
    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)


    # if args.dataset == "mnist" or args.dataset == "fmnist":
    #     generate_mnist('../dataset/mnist/', args.num_clients, 10, args.niid)
    # elif args.dataset == "Cifar10" or args.dataset == "Cifar100":
    #     generate_cifar10('../dataset/Cifar10/', args.num_clients, 10, args.niid)
    # else:
    #     generate_synthetic('../dataset/synthetic/', args.num_clients, 10, args.niid)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)

    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
