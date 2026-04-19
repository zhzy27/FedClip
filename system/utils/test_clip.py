import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
from get_clip_text_encoder import get_clip_class_embeddings
from data_utils import read_client_data

import random
import numpy as np
import torch

def set_seed(seed=42):
    """设置所有随机种子，确保结果可复现"""
    # 1. Python内置随机库
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch
    torch.manual_seed(seed)          # CPU
    torch.cuda.manual_seed(seed)      # 当前GPU
    torch.cuda.manual_seed_all(seed)  # 所有GPU

    # 4. 关闭CuDNN的自动优化算法选择（可能引入不确定性）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 警告：上述设置会降低性能（约10%-30%），但可确保可复现性
    print("固定程序随机种子")

#读取客户端数据
def load_train_data(id, dataset, batch_size=16):
    """加载训练数据"""
    train_data = read_client_data(dataset, id, is_train=True, few_shot=0)
    return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)


def load_test_data(id, dataset, batch_size=16):
    """加载测试数据"""
    test_data = read_client_data(dataset, id, is_train=False, few_shot=0)
    return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

# -------------------- 模型定义 --------------------
class ResNet18WithProjection(nn.Module):
    def __init__(self, num_classes=10, proj_dim=512):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=False)
        feat_dim = self.backbone.fc.in_features 
        # 修改第一层卷积
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 移除原模型的最大池化层
        self.backbone.maxpool = nn.Identity() 
        # 移除原来的全连接层
        self.backbone.fc = nn.Identity()

        # 投影头（将图像特征映射到 CLIP 文本嵌入空间）
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim)
        )
        # 分类头
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)  
        proj_features = self.projection(features)# [B, 512]
        logits = self.classifier(proj_features)        # [B, 10] 用于 CE 损失# [B, proj_dim] 用于对比损失
        return logits, proj_features


# -------------------- 工具函数 --------------------



# -------------------- 训练函数 --------------------
def train_epoch(model, loader, optimizer, criterion_ce,  text_features, device, alpha=1.0, use_clip=True):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_contrast = 0.0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        logits, proj_feats = model(images)

        # 1. 交叉熵损失
        loss_ce = criterion_ce(logits, labels)

        if use_clip:
            # 2. 对比损失：投影特征与对应类别的文本嵌入对齐
            # 获取当前 batch 对应的文本嵌入（根据 labels 选取）  这里传递的文本嵌入是归一化后的
            batch_text_feats = text_features[labels]  # [B, proj_dim]

            # 归一化投影特征
            proj_feats = proj_feats / proj_feats.norm(dim=-1, keepdim=True)

            # 计算余弦相似度损失（最小化 1 - cos_sim）
            cos_sim = (proj_feats * batch_text_feats).sum(dim=-1)  # [B]
            # loss_contrast = (1 - cos_sim).clamp(min=0).mean()
            loss_contrast = (1 - cos_sim).mean()


            loss = loss_ce + alpha * loss_contrast
            total_contrast += loss_contrast.item()
        else:
            loss = loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += loss_ce.item()

    avg_loss = total_loss / len(loader)
    avg_ce = total_ce / len(loader)
    avg_contrast = total_contrast / len(loader) if use_clip else 0.0
    return avg_loss, avg_ce, avg_contrast

#评估函数
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


# -------------------- 主程序 --------------------
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_clip", action="store_true", help="启用 CLIP 引导的对比损失")
    parser.add_argument("--alpha", type=float, default=1.0, help="余弦损失权重")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP 模型名称")
    parser.add_argument("--clip_model", type=str, default="RN50", help="CLIP 模型名称")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="权重衰减系数 (默认: 1e-6)")
    args = parser.parse_args()
    set_seed(0)
    device = args.device
    print(f"使用设备: {device}")
    print(f"训练模式: {'CLIP-Guided' if args.use_clip else 'Baseline (CE only)'}")

    # -------------------- 数据准备 --------------------
    trainloader = load_train_data(str(19), 'Cifar100',args.batch_size)
    testloader =load_test_data(str(19), 'Cifar100',args.batch_size)

    # -------------------- 模型与损失 --------------------
    # 获取 CLIP 文本嵌入的维度（取决于所选 CLIP 模型）

    model = ResNet18WithProjection(num_classes=100, proj_dim=1024).to(device)
    print(model)
    criterion_ce = nn.CrossEntropyLoss()
    # 如果使用 CLIP 引导，预计算所有类别的文本嵌入
    text_features = None
    if args.use_clip:
        print("预计算 CLIP 文本嵌入...")
        _,text_features = get_clip_class_embeddings('cifar100',model_name=args.clip_model ,prompt_template= "a photo of a {}",device = device)
        # 确保嵌入不参与梯度
        text_features.requires_grad_(False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay )
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -------------------- 训练循环 --------------------
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        loss, ce_loss, cont_loss = train_epoch(
            model, trainloader, optimizer,
            criterion_ce, text_features, device,
            alpha=args.alpha, use_clip=args.use_clip
        )
        scheduler.step()

        acc = evaluate(model, testloader, device)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} (CE: {ce_loss:.4f} | Cont: {cont_loss:.4f}) | Test Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"best_model_{'clip' if args.use_clip else 'baseline'}.pth")

    print(f"Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()