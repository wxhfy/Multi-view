import torch
from network import DealMVC_BYOL,MultiHeadAttention, FeedForwardNetwork
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import byol_loss
import dataloader as loader
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torch.nn.functional as F



# 设置随机种子以保证实验可重复性
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# proposed local and global contrative calibration loss
# 对比训练函数，用于模型的对比学习训练
def contrastive_train(epoch, model, data_loader, optimizer, beta):
    tot_loss = 0.0
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        xs = [x.to(device) for x in xs]  # 处理每个视图的输入数据

        # 清空梯度
        optimizer.zero_grad()

        # 计算模型前向传播结果
        online_projections, target_projections = model(xs)

        # 计算BYOL的损失
        loss = byol_loss(online_projections, target_projections)
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新在线网络参数

        # 更新目标网络参数
        model.update_target_network(beta)

        tot_loss += loss.item()

    print('Epoch {}: Total Loss {:.6f}'.format(epoch, tot_loss / len(data_loader)))


"""
The backbone network for DealMVC is MFLVC.
We adopt their original code for model training.
"""
# pretrain code from MLFVC
# 网络预训练
def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

# 创建伪标签
def make_pseudo_label(model, device):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    model.eval()
    scaler = MinMaxScaler()
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            hs, _, _, _ = model.forward(xs)
        for v in range(view):
            hs[v] = hs[v].cpu().detach().numpy()
            hs[v] = scaler.fit_transform(hs[v])

    kmeans = KMeans(n_clusters=class_num, n_init=100)
    new_pseudo_label = []
    for v in range(view):
        Pseudo_label = kmeans.fit_predict(hs[v])
        Pseudo_label = Pseudo_label.reshape(data_size, 1)
        Pseudo_label = torch.from_numpy(Pseudo_label)
        new_pseudo_label.append(Pseudo_label)

    return new_pseudo_label

def match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    new_y = torch.from_numpy(new_y).long().to(device)
    new_y = new_y.view(new_y.size()[0])
    return new_y

# 精调阶段
def fine_tuning(epoch, new_pseudo_label):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    tot_loss = 0.
    cross_entropy = torch.nn.CrossEntropyLoss()
    for batch_idx, (xs, _, idx) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, qs, _, _ = model(xs)
        loss_list = []
        for v in range(view):
            p = new_pseudo_label[v].numpy().T[0]
            with torch.no_grad():
                q = qs[v].detach().cpu()
                q = torch.argmax(q, dim=1).numpy()
                p_hat = match(p, q)
            loss_list.append(cross_entropy(qs[v], p_hat))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    if len(data_loader) == 0:
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss))
    else:
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

    # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


Dataname = 'BBCSport'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument("--temperature_f", type=float, default=0.5)
parser.add_argument("--temperature_l", type=float, default=1.0)
parser.add_argument("--threshold", type=float, default=0.8)
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--weight_decay", type=float, default=0.)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=15)
parser.add_argument("--mse_epochs", type=int, default=300)
parser.add_argument("--con_epochs", type=int, default=100)
parser.add_argument("--tune_epochs", type=int, default=50)
parser.add_argument("--feature_dim", type=int, default=512)
parser.add_argument("--high_feature_dim", type=int, default=512)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--ffn_size', type=int, default=32)
parser.add_argument('--attn_bias_dim', type=int, default=6)
parser.add_argument('--attention_dropout_rate', type=float, default=0.5)

args = parser.parse_args()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
dataset, dims, view, data_size, class_num = loader.load_data(args.dataset)

# 数据加载器设置
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)

# 设置随机种子
setup_seed(args.seed)

# 模型初始化
model = DealMVC_BYOL([dims] * view, args.feature_dim, args.high_feature_dim, view).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)