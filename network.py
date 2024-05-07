import torch
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        # Sequential container to stack layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),  # First linear layer from input dimension to 500
            nn.ReLU(),                  # ReLU activation layer
            nn.Linear(500, 500),        # Second linear layer from 500 to 500
            nn.ReLU(),                  # ReLU activation layer
            nn.Linear(500, 2000),       # Third linear layer from 500 to 2000
            nn.ReLU(),                  # ReLU activation layer
            nn.Linear(2000, feature_dim) # Final linear layer to desired feature dimension
        )

    def forward(self, x):
        return self.encoder(x)  # Forward pass through the sequential container

class ProjectionHead(nn.Module):
    def __init__(self, feature_dim, projection_dim):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        return self.projection(x)

class Predictor(nn.Module):
    def __init__(self, projection_dim):
        super(Predictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        return self.predictor(x)


class BYOL(nn.Module):
    def __init__(self, input_dims, feature_dim, projection_dim, num_heads, attention_dropout_rate, attn_bias_dim):
        super(BYOL, self).__init__()
        self.online_encoder = Encoder(input_dims, feature_dim)
        self.target_encoder = Encoder(input_dims, feature_dim)
        self.online_projection_head = ProjectionHead(feature_dim, projection_dim)
        self.online_predictor = Predictor(projection_dim)

        self.target_projection_head = ProjectionHead(feature_dim, projection_dim)

        # 添加 MultiHeadAttention
        self.attention = MultiHeadAttention(feature_dim, attention_dropout_rate, num_heads, attn_bias_dim)

        # 初始化目标网络参数与在线网络相同
        self._update_target_network(0.0)

    def _update_target_network(self, beta):
        for target_param, online_param in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            target_param.data = beta * target_param.data + (1 - beta) * online_param.data
        for target_param, online_param in zip(self.target_projection_head.parameters(),
                                              self.online_projection_head.parameters()):
            target_param.data = beta * target_param.data + (1 - beta) * online_param.data

    def forward(self, x):
        online_features = self.online_encoder(x)

        # 应用注意力机制
        online_features = self.attention(online_features.unsqueeze(1), online_features.unsqueeze(1),
                                         online_features.unsqueeze(1)).squeeze(1)

        online_projections = self.online_projection_head(online_features)
        online_predictions = self.online_predictor(online_projections)

        with torch.no_grad():  # 确保目标网络不更新梯度
            target_features = self.target_encoder(x)
            # 应用注意力机制
            target_features = self.attention(target_features.unsqueeze(1), target_features.unsqueeze(1),
                                             target_features.unsqueeze(1)).squeeze(1)
            target_projections = self.target_projection_head(target_features)

        return online_predictions, target_projections


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_bias_dim):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_bias = nn.Linear(attn_bias_dim, num_heads)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, 1)

    def forward(self, q, k, v):

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]


        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, self.num_heads * d_v)

        x = self.output_layer(x)

        return x



class FeedForwardNetwork(nn.Module):
    def __init__(self, view, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(view, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, view)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        x = x.unsqueeze(1)
        return x