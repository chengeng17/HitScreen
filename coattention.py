import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FullyConnectedNetwork(nn.Module):
    """A class for a fully connected network with optional activation and dropout."""
    def __init__(self, dims, activation='ReLU', dropout=0.0):
        super(FullyConnectedNetwork, self).__init__()

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2: 
                if activation:
                    layers.append(getattr(nn, activation)(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class CoAttentionLayer(nn.Module):
    def __init__(self, v_dim, pLMs_q_dim, h_dim, layer, activation='ReLU', dropout=0.2, K=1):
        super(CoAttentionLayer, self).__init__()

        self.v_net = FullyConnectedNetwork([v_dim, h_dim], activation=activation, dropout=dropout)
        self.q_net = FullyConnectedNetwork([pLMs_q_dim, h_dim], activation=activation, dropout=dropout)

        self.backbone = MCA_ED(layer, h_dim, dropout)


        self.att_net = nn.Linear(h_dim, K)
        self.proj_norm = LayerNorm(h_dim)


    def attention_pooling(self, v, q, att_map):
        att_map = att_map.squeeze(-1)
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        return fusion_logits

    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)

    def forward(self, v, q):
        v_mask = self.make_mask(v)
        q_mask = self.make_mask(q)

        v = self.v_net(v)
        q = self.q_net(q)


        v, q = self.backbone(v, q, v_mask, q_mask)


        att_scores = self.att_net(v.unsqueeze(2) * q.unsqueeze(1)).squeeze(-1)


        att_scores = att_scores.masked_fill(v_mask.squeeze(1).squeeze(1).unsqueeze(2).expand(-1, -1, 1200), -1e9)

        att_scores = att_scores.masked_fill(q_mask.squeeze(1).squeeze(1).unsqueeze(1).expand(-1, 300, -1), -1e9)

        att_maps = torch.softmax(att_scores, dim=-1).unsqueeze(-1)

        return v , q, att_maps

    

class MCA_ED(nn.Module):
    def __init__(self, layer, hidden_size, dropout):
        super(MCA_ED, self).__init__()
        self.layer_stack = nn.ModuleList([SGA(hidden_size, dropout) for _ in range(layer)])

    def forward(self, x, y, x_mask, y_mask):
        for layer_module in self.layer_stack:
            x, y = layer_module(x, y, x_mask, y_mask)
        return x, y

class MHAtt(nn.Module):
    def __init__(self, hidden_size, dropout, multi_head):
        super(MHAtt, self).__init__()
        hidden_size_head = int(hidden_size / multi_head)

        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.multi_head = multi_head
        self.hidden_size_head = hidden_size_head
        self.hidden_size = hidden_size

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class FFN(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout):
        super(FFN, self).__init__()
        self.network = FullyConnectedNetwork(
            dims=[hidden_size, ff_size, hidden_size],
            activation='ReLU',
            dropout=dropout
        )

    def forward(self, x):
        return self.network(x)



class SA(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SA, self).__init__()

        self.mhatt = MHAtt(hidden_size, dropout, multi_head=8)
        self.ffn = FFN(hidden_size, ff_size=hidden_size, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x

    
class SGA(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SGA, self).__init__()

        self.mhatt_self_x = MHAtt(hidden_size, dropout, multi_head=8)
        self.mhatt_self_y = MHAtt(hidden_size, dropout, multi_head=8)

        self.mhatt_cross_xy = MHAtt(hidden_size, dropout, multi_head=8)
        self.mhatt_cross_yx = MHAtt(hidden_size, dropout, multi_head=8)

        self.ffn_x = FFN(hidden_size, ff_size=hidden_size, dropout=dropout)
        self.ffn_y = FFN(hidden_size, ff_size=hidden_size, dropout=dropout)

        self.dropout1_x = nn.Dropout(dropout)
        self.norm1_x = LayerNorm(hidden_size)
        self.dropout2_x = nn.Dropout(dropout)
        self.norm2_x = LayerNorm(hidden_size)
        self.dropout3_x = nn.Dropout(dropout)
        self.norm3_x = LayerNorm(hidden_size)

        self.dropout1_y = nn.Dropout(dropout)
        self.norm1_y = LayerNorm(hidden_size)
        self.dropout2_y = nn.Dropout(dropout)
        self.norm2_y = LayerNorm(hidden_size)
        self.dropout3_y = nn.Dropout(dropout)
        self.norm3_y = LayerNorm(hidden_size)

    def forward(self, x, y, x_mask, y_mask):

        x = self.norm1_x(x + self.dropout1_x(
            self.mhatt_self_x(x, x, x, x_mask)
        ))

        y = self.norm1_y(y + self.dropout1_y(
            self.mhatt_self_y(y, y, y, y_mask)
        ))

        x = self.norm2_x(x + self.dropout2_x(
            self.mhatt_cross_xy(y, y, x, y_mask)
        ))

        y = self.norm2_y(y + self.dropout2_y(
            self.mhatt_cross_yx(x, x, y, x_mask)
        ))

        x = self.norm3_x(x + self.dropout3_x(
            self.ffn_x(x)
        ))

        y = self.norm3_y(y + self.dropout3_y(
            self.ffn_y(y)
        ))

        return x, y
    


