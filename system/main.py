#!/usr/bin/env python

import os
import sys
if "-did" in sys.argv:
    idx = sys.argv.index("-did")
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[idx+1]
elif "--device_id" in sys.argv:
    idx = sys.argv.index("--device_id")
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[idx+1]

def _get_int_arg(flag_short, flag_long, default):
    for flag in (flag_short, flag_long):
        if flag in sys.argv:
            idx = sys.argv.index(flag)
            if idx + 1 < len(sys.argv):
                try:
                    return int(sys.argv[idx + 1])
                except ValueError:
                    return default
    return default

_clip_cpu_threads = _get_int_arg("-clip_cpu_threads", "--clip_cpu_threads", 4)
if _clip_cpu_threads > 0:
    os.environ.setdefault("OMP_NUM_THREADS", str(_clip_cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(_clip_cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_clip_cpu_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_clip_cpu_threads))
import torch
import argparse
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
from flcore.servers.serverre import FedRE
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
import random
#日志文件
# def set_seed(seed=0):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     # 必须加上这两行！关闭 CuDNN 的自动调优，强制使用确定性算法
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     # 在某些 PyTorch 版本中，还需要强制底层算子确定性
#     torch.use_deterministic_algorithms(True)
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def set_seed(seed=0):
    """
    固定宏观数据划分和初始化的随机性，但释放底层 CuDNN 算子的性能。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True   
    torch.backends.cudnn.deterministic = False 

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0) 
# set_seed(0)

def run(args):

    time_list = []
    reporter = MemReporter()
    args.save_file_paths = []
    input_size = 32
    if "Tiny" in args.dataset:
        input_size = 64


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
                'FedAvgCNN_Hetero_AFM(in_features=3, num_classes=args.num_classes, dim=2000, ratio_LR=1.0)'  
            ]
            args.global_model = 'FedAvgCNN_Homo_AFM(in_features=3, num_classes=args.num_classes, dim=1600, ratio_LR=1.0)'
        elif args.model_family == "SPU_CNN1":
            args.models = [
                'CNN_1_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)'  
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
                'CNN_1_512_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)'  
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
            args.global_model ='low_rank_resnet18_cifar(features=[64, 128, 256, 512],num_classes=args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation = None,norm_layer=None, has_norm = False,bn_block_num = 4, ratio_LR = 1.0)'
        elif args.model_family == "low_rank_resnet_5_bn":
            args.models = [
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = True,bn_block_num = 4, ratio_LR = 1.0)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = True,bn_block_num = 4, ratio_LR = 0.4)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = True,bn_block_num = 4, ratio_LR = 0.2)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = True,bn_block_num = 4, ratio_LR = 0.12)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,has_norm = True,bn_block_num = 4, ratio_LR = 0.05)',         
            ]
            args.global_model ='low_rank_resnet18_cifar(features=[64, 128, 256, 512],num_classes=args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation = None,norm_layer=None, has_norm = True,bn_block_num = 4, ratio_LR = 1.0)'
        elif args.model_family == "low_rank_resnet_5_in":
            args.models = [
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=instance_norm,has_norm = True,bn_block_num = 4, ratio_LR = 1.0)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=instance_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.4)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=instance_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.2)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=instance_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.12)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=instance_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.05)',         
            ]
            args.global_model ='low_rank_resnet18_cifar(features=[64, 128, 256, 512],num_classes=args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation = None,norm_layer=instance_norm, has_norm = True,bn_block_num = 4, ratio_LR = 1.0)'
        elif args.model_family == "low_rank_resnet_5_gn":
            args.models = [
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=group_norm,has_norm = True,bn_block_num = 4, ratio_LR = 1.0)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=group_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.4)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=group_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.2)',
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=group_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.12)',    
                'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=group_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.05)',                      
            ]
            args.global_model ='low_rank_resnet18_cifar(features=[64, 128, 256, 512],num_classes=args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation = None,norm_layer=group_norm, has_norm = True,bn_block_num = 4, ratio_LR = 1.0)'
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
                f'Hyper_CNN_512(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.9, input_size = {input_size})', # 暂时只考虑一个秩
                f'Hyper_CNN_512(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.37, input_size = {input_size})',
                f'Hyper_CNN_512(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.35, input_size = {input_size})',
                f'Hyper_CNN_512(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.25, input_size = {input_size})',
                f'Hyper_CNN_512(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.15, input_size = {input_size})',
            ]
            args.global_model = f'Hyper_CNN_512(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.15, input_size = {input_size})'
        elif args.model_family == "CNN-512":
            args.models = [
                f'CNN_512(in_channels=3, n_kernels=16, out_dim=args.num_classes, input_size = {input_size})',
            ]
            args.global_model = f'CNN_512(in_channels=3, n_kernels=16, out_dim=args.num_classes, input_size = {input_size})'
        elif args.model_family == "CNN-5-512":
            args.models = [
                'CNN_1_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_2_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_3_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_4_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_5_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)', 
            ]
            args.global_model = 'CNN_5_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)' 
        elif args.model_family == "CNN-5-512-tiny":
            args.models = [
                'CNN_1_512_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_2_512_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_3_512_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_4_512_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_5_512_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)', 
            ]
            args.global_model = 'CNN_5_512_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)'
        elif args.model_family == "CNN-5-512-AFM-tiny":
            args.models = [
                'CNN_1_hetero_AFM_512_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_2_hetero_AFM_512_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_3_hetero_AFM_512_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_4_hetero_AFM_512_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_5_hetero_AFM_512_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
            ]
            args.global_model = 'CNN_5_homo_AFM_512_tiny(in_channels=3, n_kernels=16, out_dim=args.num_classes)'     
        elif args.model_family == "CNN-5-512-AFM":
            args.models = [
                'CNN_1_hetero_AFM_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_2_hetero_AFM_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_3_hetero_AFM_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_4_hetero_AFM_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
                'CNN_5_hetero_AFM_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
            ]
            args.global_model = 'CNN_5_homo_AFM_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)'
        elif args.model_family in ["ResNet18-5-AFM", "ResNet18-5"]:
            resnet18_widths = [64, 56, 48, 40, 32]
            resnet18_factory = "resnet18_afm" if args.model_family == "ResNet18-5-AFM" else "resnet18_family"
            args.models = [
                f'{resnet18_factory}(in_channels=3, num_classes=args.num_classes, base_width={width}, input_size={input_size}, feature_dim=args.feature_dim)'
                for width in resnet18_widths
            ]
            args.global_model = f'{resnet18_factory}(in_channels=3, num_classes=args.num_classes, base_width={min(resnet18_widths)}, input_size={input_size}, feature_dim=args.feature_dim)'
        elif args.model_family == "CNN-512-AFM":
            args.models = [
                'CNN_1_hetero_AFM_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)',
            ]
            args.global_model = 'CNN_homo_AFM_512(in_channels=3, n_kernels=16, out_dim=args.num_classes)'    
        elif args.model_family == "Decom_resnet18_5":
            args.models = [
                f'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=layer_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.5, input_size = {input_size}, rank_dropout_mode=args.rank_dropout_mode, rank_dropout_stage_start=args.rank_dropout_stage_start, rank_dropout_stage_end=args.rank_dropout_stage_end)',
                f'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=layer_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.4, input_size = {input_size}, rank_dropout_mode=args.rank_dropout_mode, rank_dropout_stage_start=args.rank_dropout_stage_start, rank_dropout_stage_end=args.rank_dropout_stage_end)',
                f'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=layer_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.29, input_size = {input_size}, rank_dropout_mode=args.rank_dropout_mode, rank_dropout_stage_start=args.rank_dropout_stage_start, rank_dropout_stage_end=args.rank_dropout_stage_end)',
                f'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=layer_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.2, input_size = {input_size}, rank_dropout_mode=args.rank_dropout_mode, rank_dropout_stage_start=args.rank_dropout_stage_start, rank_dropout_stage_end=args.rank_dropout_stage_end)',
                f'low_rank_resnet18_cifar(features= [64, 128, 256, 512],num_classes = args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=layer_norm,has_norm = True,bn_block_num = 4, ratio_LR = 0.12, input_size = {input_size}, rank_dropout_mode=args.rank_dropout_mode, rank_dropout_stage_start=args.rank_dropout_stage_start, rank_dropout_stage_end=args.rank_dropout_stage_end)',
            ]
            args.global_model = f'low_rank_resnet18_cifar(features=[64, 128, 256, 512],num_classes=args.num_classes,zero_init_residual = False,groups= 1,width_per_group=64,replace_stride_with_dilation = None,norm_layer=layer_norm, has_norm = True,bn_block_num = 4, ratio_LR = 1.0, input_size = {input_size}, rank_dropout_mode=args.rank_dropout_mode, rank_dropout_stage_start=args.rank_dropout_stage_start, rank_dropout_stage_end=args.rank_dropout_stage_end)'
        elif args.model_family in ["SPU_ResNet18_1"]:
            resnet18_widths = 64
            resnet18_factory = "resnet18_family"
            args.models = [
                f'{resnet18_factory}(in_channels=3, num_classes=args.num_classes, base_width={resnet18_widths}, input_size={input_size}, feature_dim=args.feature_dim)'
            ]
            args.global_model = f'{resnet18_factory}(in_channels=3, num_classes=args.num_classes, base_width={resnet18_widths}, input_size={input_size}, feature_dim=args.feature_dim)'
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

        elif args.algorithm == "FedRE":
            server = FedRE(args, i)

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
    average_data(args.save_file_paths)

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
    parser.add_argument("--final-model-root", type=str, default="./final_models",
                        help="Folder for overwritten final model snapshots grouped by dataset/algorithm/model/partition.")
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
    # FedRE
    parser.add_argument('-re_samples', "--re_samples", type=int, default=1,
                        help="Number of entangled representations uploaded by each selected client.")
    parser.add_argument('-hbs', "--head_batch_size", type=int, default=10,
                        help="Batch size for FedRE server head training.")
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
    parser.add_argument('-v_mse_lamda', "--v_mse_lamda", type=float, default=0.0, help="clip vision loss")
    # 聚合部分的几个超参数
    parser.add_argument('-aggregate_tau', "--aggregate_tau", type=float, default=1.0, help="Aggregate function temperature")
    parser.add_argument('-aggregate_power', "--aggregate_power", type=float, default=0.0, help="Power of the Aggregate Function")
    parser.add_argument('-aggregate_gamma', "--aggregate_gamma", type=float, default=0.0, help="Self-protection of aggregation functions")
    parser.add_argument('-anchor_tau', "--anchor_tau", type=float, default=1.0, help="anchor loss tau")
    parser.add_argument('-u_lr_ratio', "--u_lr_ratio", type=float, default=0.1, help="Learning-rate ratio for low-rank U parameters in FedCLIP ResNet; unused by CNN FedCLIP")
    parser.add_argument("--rank_dropout_mode", type=str, default="dynamic_capacity",
                        choices=["dynamic_capacity", "original", "capacity", "none"],
                        help="FedCLIP ResNet low-rank dropout mode; unused by CNN FedCLIP")
    parser.add_argument("--rank_dropout_stage_start", type=float, default=0.3,
                        help="Progress ratio where dynamic_capacity starts moving from full rank to capacity-aware dropout")
    parser.add_argument("--rank_dropout_stage_end", type=float, default=0.8,
                        help="Progress ratio where dynamic_capacity becomes fully capacity-aware dropout")
    parser.add_argument("--h5_result_root", type=str, default="./h5_results",
                        help="Structured root directory for H5 convergence/result files")
    parser.add_argument('-clip_cpu_threads', "--clip_cpu_threads", type=int, default=4, help="Max CPU threads used by FedCLIP CLIP-anchor helpers; set 0 to disable")
    parser.add_argument("--aggregation_mode", type=str, default=None,
                        choices=["avg", "delta_avg", "projection", "consensus_projection", "sign_personalized_projection", "sign_projection_norm_restore", "sign_projection_no_group_renorm", "sign_projection_weight", "coeff_lowrank_sparse"],
                        help="FedCLIP CNN aggregation mode. If omitted, the legacy use_common_residual_projection flag is used.")
    parser.add_argument("--use_common_residual_projection", type=int, default=1,
                        help="FedCLIP CNN aggregation. 1 enables common-residual projection after warm-up; 0 uses plain sample-size FedAvg.")
    parser.add_argument("--projection_warmup_ratio", type=float, default=0.2,
                        help="Warm-up ratio for FedCLIP CNN projection aggregation. During warm-up, plain FedAvg is used.")
    parser.add_argument("--projection_energy", type=float, default=0.8,
                        help="Energy threshold for adaptive common subspace dimension.")
    parser.add_argument("--projection_k_max", type=int, default=5,
                        help="Maximum common subspace dimension per layer.")
    parser.add_argument("--projection_layer_scope", type=str,
                        choices=["low_rank", "low_rank_plus_classifier", "all_weight"],
                        default="low_rank",
                        help="Projected layer range for sign_projection_no_group_renorm: low-rank layers only, plus the final classifier, or all matrix weights. Other aggregation modes ignore this option.")
    parser.add_argument("--personalized_rank_selection", type=int, choices=[0, 1], default=0,
                        help="For sign_projection_no_group_renorm/sign_projection_weight: 1 enables per-client SVD direction selection; 0 keeps the shared top-K directions.")
    parser.add_argument("--personalized_rank_num", type=int, default=5,
                        help="Per-client direction count M in personalized_rank_mode=fixed; ignored in energy mode. Whether direction 0 is retained is controlled by personalized_rank_force_u1.")
    parser.add_argument("--personalized_rank_force_u1", type=int, choices=[0, 1], default=1,
                        help="Per-client rank selection: 1 starts with direction 0, then selects remaining directions by score (fixed) or accumulates their energy (energy); 0 freely selects from all valid directions.")
    parser.add_argument("--personalized_rank_mode", type=str, choices=["fixed", "energy"], default="fixed",
                        help="Per-client direction selection mode: fixed uses personalized_rank_num; energy uses the smallest per-client set reaching personalized_rank_energy.")
    parser.add_argument("--personalized_rank_energy", type=float, default=0.8,
                        help="Per-client cumulative direction-energy threshold tau in personalized_rank_mode=energy; must be in (0, 1].")
    parser.add_argument("--personalized_direction_selection_mode", type=str,
                        choices=["delta", "model_only", "model_delta_joint", "joint_transfer"], default="delta",
                        help="Direction-selection source: delta preserves the current implementation; model_only/model_delta_joint use the local-start model; joint_transfer jointly selects a cross-layer source-client subset and per-layer directions and requires joint_subset_opt.")
    parser.add_argument("--personalized_extra_topk", type=int, default=1,
                        help="Number of valid tail directions selected by model_only/model_delta_joint in addition to the always-retained direction 0; must be non-negative.")
    parser.add_argument("--personalized_cross_layer_client_mode", type=str,
                        choices=["none", "consensus_topk", "all_direction_topk", "joint_subset_opt"], default="none",
                        help="Optional cross-layer source-client constraint. consensus_topk scores selected tail directions; all_direction_topk scores every tail direction; joint_subset_opt jointly optimizes the shared source-client subset and per-layer directions and requires joint_transfer.")
    parser.add_argument("--personalized_cross_layer_client_topk", type=int, default=5,
                        help="Maximum number of external collaborator clients retained per target by consensus_topk/all_direction_topk/joint_subset_opt; must be in [1, num_clients].")
    parser.add_argument("--personalized_g_scale", type=int, choices=[0, 1], default=1,
                        help="After personalized selection, 1 keeps the existing g scaling; 0 uses g only through direction scores and does not scale selected coefficients by g.")
    parser.add_argument("--local_update_views", type=int, choices=[1, 2], default=1,
                        help="Number of independent local-update views. 2 trains A and diagnostic B from the same received model; aggregation still uses A only.")
    parser.add_argument("--personalized_repeatability_threshold", type=float, default=-1.0,
                        help="Normalized A/B direction-repeatability threshold in [-1, 1]. -1 disables filtering; values above -1 require two local-update views and energy rank selection.")
    parser.add_argument("--personalized_coeff_mode", type=str,
                        choices=["same_sign", "self", "avg"], default="same_sign",
                        help="Coefficient used on selected personalized directions. same_sign preserves the current aggregation; self and avg are diagnostic ablations.")
    parser.add_argument("--personalized_m_filter_mode", type=str,
                        choices=["none", "dominant_side"], default="none",
                        help="Optional second-stage filter for the raw personalized direction set. dominant_side keeps only clients on a direction's sample-weighted energy-dominant sign side.")
    parser.add_argument("--personalized_dominance_threshold", type=float, default=0.7,
                        help="Minimum one-side energy dominance ratio P_k required by dominant_side (0.5 < threshold <= 1). A direction passes only when P_k >= threshold and the target client is on the dominant-sign side. Examples: 0.6/0.7/0.8 allow the weak side at most about 40%/30%/20% energy, respectively.")
    parser.add_argument("--personalized_conflict_handling", type=str,
                        choices=["zero", "self"], default="zero",
                        help="How dominant_side handles raw-selected directions that fail shared routing: zero drops them (legacy behavior); self keeps the target client's own signed coefficient without alpha weighting or same-sign aggregation. Ignored when personalized_m_filter_mode=none.")
    parser.add_argument("--personalized_tail_scale", type=float, default=1.0,
                        help="Scale lambda for selected directions after direction 0. 1 preserves the full update; 0 keeps the K=1 base when direction 0 is retained.")
    parser.add_argument("--projection_norm_scale_max", type=float, default=2.0,
                        help="Maximum client-wise norm restoration scale for sign projection norm-restore modes.")
    parser.add_argument("--projection_use_residual", type=int, default=1,
                        help="1 keeps client residuals; 0 only uses the common projected update.")
    parser.add_argument("--projection_residual_ema", type=int, default=0,
                        help="1 uses EMA historical residuals; 0 uses only current-round residual compensation.")
    parser.add_argument("--coeff_rho_c", type=float, default=0.1,
                        help="Relative nuclear-norm threshold for the collaborative coefficient matrix in coeff_lowrank_sparse.")
    parser.add_argument("--coeff_rho_p", type=float, default=0.05,
                        help="Relative element-wise sparse threshold for the personalized coefficient matrix in coeff_lowrank_sparse.")
    parser.add_argument("--coeff_decomp_iters", type=int, default=15,
                        help="Maximum alternating proximal iterations for coeff_lowrank_sparse.")
    parser.add_argument("--coeff_decomp_tol", type=float, default=1e-5,
                        help="Relative stopping tolerance for coeff_lowrank_sparse.")
    parser.add_argument("--coeff_decomp_warmup_ratio", type=float, default=0.2,
                        help="FedAvg warm-up ratio before coeff_lowrank_sparse is enabled.")
    parser.add_argument("--personal_residual_beta", type=float, default=0.1,
                        help="Current-round residual coefficient when projection_residual_ema=0.")
    parser.add_argument("--personal_residual_mu", type=float, default=0.9,
                        help="EMA coefficient for client personalized residuals.")
    parser.add_argument("--personal_residual_gamma", type=float, default=0.5,
                        help="Residual update strength for client personalized residuals.")
    parser.add_argument("--personal_residual_clip", type=float, default=0.0,
                        help="Residual norm clip ratio relative to current global weight. 0 disables clipping.")

    args = parser.parse_args()

    if not np.isfinite(args.projection_energy) or not (
        0.0 < args.projection_energy <= 1.0
    ):
        parser.error("--projection_energy must be in the interval (0, 1].")
    if not np.isfinite(args.coeff_rho_c) or args.coeff_rho_c < 0.0:
        parser.error("--coeff_rho_c must be finite and non-negative.")
    if not np.isfinite(args.coeff_rho_p) or args.coeff_rho_p < 0.0:
        parser.error("--coeff_rho_p must be finite and non-negative.")
    if args.coeff_decomp_iters < 1:
        parser.error("--coeff_decomp_iters must be at least 1.")
    if (
        not np.isfinite(args.coeff_decomp_tol)
        or args.coeff_decomp_tol <= 0.0
    ):
        parser.error("--coeff_decomp_tol must be finite and greater than 0.")
    if (
        not np.isfinite(args.coeff_decomp_warmup_ratio)
        or not 0.0 <= args.coeff_decomp_warmup_ratio <= 1.0
    ):
        parser.error("--coeff_decomp_warmup_ratio must be in [0, 1].")
    if (
        args.personalized_rank_selection
        and args.personalized_direction_selection_mode == "delta"
        and args.personalized_rank_mode == "fixed"
        and args.personalized_rank_num < 1
    ):
        parser.error("--personalized_rank_num must be at least 1.")
    if args.personalized_rank_mode == "energy" and (
        not np.isfinite(args.personalized_rank_energy)
        or not 0.0 < args.personalized_rank_energy <= 1.0
    ):
        parser.error("--personalized_rank_energy must be in the interval (0, 1].")
    if args.personalized_extra_topk < 0:
        parser.error("--personalized_extra_topk must be non-negative.")
    if args.personalized_cross_layer_client_topk < 1:
        parser.error(
            "--personalized_cross_layer_client_topk must be at least 1."
        )
    if (
        args.personalized_cross_layer_client_mode != "none"
        and args.personalized_cross_layer_client_topk > args.num_clients
    ):
        parser.error(
            "--personalized_cross_layer_client_topk must satisfy "
            "1 <= topk <= num_clients."
        )
    if args.personalized_cross_layer_client_mode == "consensus_topk" and (
        args.aggregation_mode != "sign_projection_no_group_renorm"
        or not args.personalized_rank_selection
        or args.personalized_coeff_mode != "same_sign"
        or args.personalized_m_filter_mode != "none"
        or args.personalized_conflict_handling != "zero"
    ):
        parser.error(
            "--personalized_cross_layer_client_mode consensus_topk requires "
            "sign_projection_no_group_renorm, personalized_rank_selection=1, "
            "same_sign coefficients, personalized_m_filter_mode=none, and "
            "personalized_conflict_handling=zero."
        )
    if args.personalized_cross_layer_client_mode == "all_direction_topk" and (
        args.aggregation_mode != "sign_projection_no_group_renorm"
        or args.personalized_direction_selection_mode != "delta"
        or not args.personalized_rank_selection
        or args.personalized_coeff_mode != "same_sign"
        or args.personalized_m_filter_mode != "none"
        or args.personalized_conflict_handling != "zero"
    ):
        parser.error(
            "--personalized_cross_layer_client_mode all_direction_topk "
            "requires sign_projection_no_group_renorm, "
            "personalized_direction_selection_mode=delta, "
            "personalized_rank_selection=1, same_sign coefficients, "
            "personalized_m_filter_mode=none, and "
            "personalized_conflict_handling=zero."
        )
    joint_transfer_enabled = (
        args.personalized_direction_selection_mode == "joint_transfer"
    )
    joint_subset_enabled = (
        args.personalized_cross_layer_client_mode == "joint_subset_opt"
    )
    if joint_transfer_enabled != joint_subset_enabled:
        parser.error(
            "personalized_direction_selection_mode=joint_transfer and "
            "personalized_cross_layer_client_mode=joint_subset_opt must be "
            "enabled together."
        )
    if joint_transfer_enabled and (
        args.aggregation_mode != "sign_projection_no_group_renorm"
        or not args.personalized_rank_selection
        or args.personalized_rank_mode != "fixed"
        or args.personalized_rank_num < 1
        or args.personalized_rank_force_u1 != 0
        or args.personalized_coeff_mode != "same_sign"
        or args.personalized_m_filter_mode != "none"
        or args.personalized_conflict_handling != "zero"
        or args.personalized_repeatability_threshold > -1.0
        or args.personalized_tail_scale != 1.0
    ):
        parser.error(
            "joint_transfer + joint_subset_opt requires "
            "sign_projection_no_group_renorm, personalized_rank_selection=1, "
            "personalized_rank_mode=fixed, personalized_rank_num>=1, "
            "personalized_rank_force_u1=0, personalized_coeff_mode=same_sign, "
            "personalized_m_filter_mode=none, "
            "personalized_conflict_handling=zero, and repeatability filtering "
            "disabled, with personalized_tail_scale=1."
        )
    if args.personalized_direction_selection_mode in {
        "model_only",
        "model_delta_joint",
    } and (
        args.aggregation_mode != "sign_projection_no_group_renorm"
        or not args.personalized_rank_selection
        or args.personalized_coeff_mode != "same_sign"
        or args.personalized_repeatability_threshold > -1.0
    ):
        parser.error(
            "model_only/model_delta_joint require "
            "--aggregation_mode sign_projection_no_group_renorm, "
            "--personalized_rank_selection 1, "
            "--personalized_coeff_mode same_sign, and repeatability filtering "
            "disabled."
        )
    if (
        not np.isfinite(args.personalized_repeatability_threshold)
        or not -1.0 <= args.personalized_repeatability_threshold <= 1.0
    ):
        parser.error(
            "--personalized_repeatability_threshold must be in [-1, 1]."
        )
    if args.personalized_repeatability_threshold > -1.0 and (
        args.local_update_views != 2
        or not args.personalized_rank_selection
        or args.personalized_rank_mode != "energy"
    ):
        parser.error(
            "Repeatability filtering requires --local_update_views 2, "
            "--personalized_rank_selection 1, and "
            "--personalized_rank_mode energy."
        )
    if (
        args.personalized_coeff_mode != "same_sign"
        and not args.personalized_rank_selection
    ):
        parser.error(
            "--personalized_coeff_mode self/avg requires "
            "--personalized_rank_selection 1."
        )
    if not np.isfinite(args.personalized_dominance_threshold):
        parser.error("--personalized_dominance_threshold must be finite.")
    if (
        args.personalized_direction_selection_mode == "delta"
        and args.personalized_m_filter_mode == "dominant_side"
        and not (
            0.5 < args.personalized_dominance_threshold <= 1.0
        )
    ):
        parser.error(
            "--personalized_dominance_threshold must satisfy 0.5 < threshold "
            "<= 1.0 when --personalized_m_filter_mode dominant_side."
        )
    if (
        args.personalized_direction_selection_mode == "delta"
        and args.personalized_m_filter_mode == "none"
        and not 0.0 <= args.personalized_dominance_threshold <= 1.0
    ):
        parser.error("--personalized_dominance_threshold must be in [0, 1].")
    if (
        args.personalized_direction_selection_mode == "delta"
        and args.personalized_m_filter_mode == "dominant_side"
        and (
            args.aggregation_mode != "sign_projection_no_group_renorm"
            or not args.personalized_rank_selection
            or args.personalized_rank_mode != "energy"
            or args.personalized_coeff_mode != "same_sign"
        )
    ):
        parser.error(
            "--personalized_m_filter_mode dominant_side requires "
            "--aggregation_mode sign_projection_no_group_renorm, "
            "--personalized_rank_selection 1, "
            "--personalized_rank_mode energy, and "
            "--personalized_coeff_mode same_sign."
        )
    if (
        not np.isfinite(args.personalized_tail_scale)
        or args.personalized_tail_scale < 0.0
    ):
        parser.error("--personalized_tail_scale must be finite and non-negative.")
    if args.personalized_tail_scale != 1.0 and (
        not args.personalized_rank_selection
        or (
            args.personalized_direction_selection_mode == "delta"
            and not args.personalized_rank_force_u1
        )
    ):
        parser.error(
            "Tail-scale ablation requires --personalized_rank_selection 1 "
            "and --personalized_rank_force_u1 1 so the K=1 base exists."
        )
    if (
        args.personalized_tail_scale != 1.0
        and args.personalized_repeatability_threshold > -1.0
    ):
        parser.error(
            "Tail-scale and repeatability-filter ablations must be run "
            "separately because filtering can remove the required K=1 base."
        )
    if (
        args.personalized_rank_selection
        and args.aggregation_mode not in {
            "sign_projection_no_group_renorm",
            "sign_projection_weight",
        }
    ):
        parser.error(
            "--personalized_rank_selection 1 requires "
            "--aggregation_mode sign_projection_no_group_renorm or "
            "sign_projection_weight."
        )

    if args.clip_cpu_threads > 0:
        torch.set_num_threads(args.clip_cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(args.clip_cpu_threads, 4)))
        except RuntimeError:
            pass

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
