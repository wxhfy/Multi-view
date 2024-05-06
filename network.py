import torch
import torch.nn as nn
from torch.nn.functional import normalize
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)



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


class DealMVC_BYOL(nn.Module):
    def __init__(self, input_dims, feature_dim, projection_dim, view_count):
        super(DealMVC_BYOL, self).__init__()
        self.view_count = view_count
        self.online_encoders = nn.ModuleList([Encoder(input_dim, feature_dim) for input_dim in input_dims])
        self.target_encoders = nn.ModuleList([Encoder(input_dim, feature_dim) for input_dim in input_dims])
        self.online_projections = nn.ModuleList(
            [ProjectionHead(feature_dim, projection_dim) for _ in range(view_count)])
        self.target_projections = nn.ModuleList(
            [ProjectionHead(feature_dim, projection_dim) for _ in range(view_count)])

        self._update_target_network(0.0)  # Initially match target network with online network

    def _update_target_network(self, beta):
        for online, target in zip(self.online_encoders, self.target_encoders):
            for online_param, target_param in zip(online.parameters(), target.parameters()):
                target_param.data = beta * target_param.data + (1 - beta) * online_param.data
        for online, target in zip(self.online_projections, self.target_projections):
            for online_param, target_param in zip(online.parameters(), target.parameters()):
                target_param.data = beta * target_param.data + (1 - beta) * online_param.data

    def forward(self, xs):
        online_features = [encoder(x) for encoder, x in zip(self.online_encoders, xs)]
        target_features = [encoder(x) for encoder, x in zip(self.target_encoders, xs)]

        online_projections = [proj(feat) for proj, feat in zip(self.online_projections, online_features)]
        target_projections = [proj(feat) for proj, feat in zip(self.target_projections, target_features)]

        return online_projections, target_projections


