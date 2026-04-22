import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
import argparse


#FedARA
# random.seed(1)
# np.random.seed(1)
#FedDAR
random.seed(0)
np.random.seed(0)
num_clients = 20
dir_path = "Cifar100/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition, alpha, cpc):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition, alpha=alpha, class_per_client=cpc):
        return
        
    # Get Cifar100 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=cpc, alpha=alpha)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, alpha=alpha, class_per_client=cpc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--niid', type=int, default=1, help="0: IID, 1: Non-IID")
    parser.add_argument('--balance', type=int, default=0, help="0: unbalanced, 1: balanced")
    parser.add_argument('--partition', type=str, default='dir', choices=['dir', 'pat', 'exdir', '-'])
    parser.add_argument('--alpha', type=float, default=0.1, help="Dirichlet distribution coefficient")
    parser.add_argument('--cpc', type=int, default=6, help="Classes per client for pathological distribution")
    args = parser.parse_args()

    niid = args.niid == 1
    balance = args.balance == 1
    partition = args.partition if args.partition != "-" else None

    # 动态构建隔离目录
    if not niid:
        sub_dir = "iid"
    elif partition == "dir":
        sub_dir = f"dir_{args.alpha}"
    elif partition == "pat":
        sub_dir = f"pat_{args.cpc}"
    elif partition == "exdir":
        sub_dir = f"exdir_{args.cpc}_{args.alpha}"
    else:
        sub_dir = str(partition)

    dir_path = f"Cifar100/{sub_dir}/"
    print(f"数据将生成在: dataset/{dir_path}")

    # 调用时传入额外的超参数
    # 注意：你需要确保 generate_dataset 和 separate_data 都能把 alpha 传下去
    generate_dataset(dir_path, num_clients, niid, balance, partition, args.alpha, args.cpc)