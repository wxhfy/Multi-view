import torch.nn.functional as F


def byol_loss(online_projection, target_projection):
    """
    计算BYOL的损失函数，目标是最小化在线网络和目标网络输出的平方L2范数距离。
    :param online_projection: 在线网络的投影输出
    :param target_projection: 目标网络的投影输出
    :return: 损失值
    """
    # 对投影向量进行归一化处理
    online_projection = F.normalize(online_projection, dim=1)
    target_projection = F.normalize(target_projection, dim=1)

    # 最小化它们的距离
    loss = 2 - 2 * (online_projection * target_projection).sum(dim=1)
    return loss.mean()


