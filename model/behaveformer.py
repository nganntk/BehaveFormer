"""BehaveFormer classes"""
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, gre_k, feature_dim, seq_len, imu_type):
        super().__init__()
        
        self.embedding = nn.Parameter(torch.zeros([gre_k, feature_dim], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.positions = torch.tensor([i for i in range(seq_len)], requires_grad=False).unsqueeze(1).repeat(1, gre_k)
        s = 0.0
        interval = seq_len / gre_k
        mu = []
        for _ in range(gre_k):
            mu.append(nn.Parameter(torch.tensor(s, dtype=torch.float), requires_grad=True))
            s = s + interval
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float).unsqueeze(0), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor([torch.tensor([50.0], dtype=torch.float, requires_grad=True) for _ in range(gre_k)]).unsqueeze(0))
        
        self.imu_type = imu_type

    def normal_pdf(self, pos, mu, sigma):
        a = pos - mu
        log_p = -1*torch.mul(a, a)/(2*(sigma**2)) - torch.log(sigma)
        return torch.nn.functional.softmax(log_p, dim=1)

    def forward(self, inputs):
        pdfs = self.normal_pdf(self.positions, self.mu, self.sigma)
        pos_enc = torch.matmul(pdfs, self.embedding)
        if self.imu_type != 'none':
            return inputs + pos_enc.unsqueeze(0).repeat(inputs.size(0), 1, 1)
        else:
            return inputs + pos_enc.unsqueeze(0)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, feature_dim, temporal_heads, channel_heads, dropout, seq_len):
        super(TransformerEncoderLayer, self).__init__()
        
        self.temporal_attention = nn.MultiheadAttention(feature_dim, temporal_heads, batch_first=True)
        self.channel_attention = nn.MultiheadAttention(seq_len, channel_heads, batch_first=True)
        
        self.attn_norm = nn.LayerNorm(feature_dim)
        
        self.cnn_units = 1
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.cnn_units, (1, 1)),
            nn.BatchNorm2d(self.cnn_units),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv2d(self.cnn_units, self.cnn_units, (3, 3), padding=1),
            nn.BatchNorm2d(self.cnn_units),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv2d(self.cnn_units, 1, (5, 5), padding=2),
            nn.BatchNorm2d(1),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
        self.final_norm = nn.LayerNorm(feature_dim)

    def forward(self, src, src_mask=None):
        src = self.attn_norm(src + self.temporal_attention(src, src, src)[0] + self.channel_attention(src.transpose(-1, -2), src.transpose(-1, -2), src.transpose(-1, -2))[0].transpose(-1, -2))
        src = self.final_norm(src + self.cnn(src.unsqueeze(dim=1)).squeeze(dim=1))
            
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim, temporal_heads, channel_heads, seq_len, num_layer, dropout):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(TransformerEncoderLayer(feature_dim, temporal_heads, channel_heads, dropout, seq_len))

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)

        return src

class Transformer(nn.Module):
    def __init__(self, num_layer, feature_dim, gre_k, temporal_heads, channel_heads, seq_len, dropout, imu_type):
        super(Transformer, self).__init__()

        self.pos_encoding = PositionalEncoding(gre_k, feature_dim, seq_len, imu_type)

        self.encoder = TransformerEncoder(feature_dim, temporal_heads, channel_heads, seq_len, num_layer, dropout)

    def forward(self, inputs):
        encoded_inputs = self.pos_encoding(inputs)

        return self.encoder(encoded_inputs)

class BehaveFormer(nn.Module):
    def __init__(self, 
                 behave_feature_dim: int,       # e.g., 8 for scroll
                 imu_feature_dim: int,          # e.g., 36 for using all imu
                 behave_len: int,               # e.g., 50 for scroll
                 imu_len: int,                  # e.g., 100
                 target_len: int,               # e.g., 64
                 gre_k: int,                    # e.g., 20
                 behave_temporal_heads: int,    # e.g., 4 (scroll_channel_heads)
                 behave_channel_heads: int,     # e.g., 10
                 imu_temporal_heads: int,       # e.g., 6, same as original code
                 imu_channel_heads: int,        # e.g., 10 
                 imu_type: str='none',
                 num_layer: int=5,
                 dropout_enc: float=0.1,
                 dropout_linear: float=0.1):
        super(BehaveFormer, self).__init__()

        self.behave_transformer = Transformer(num_layer, behave_feature_dim, gre_k, behave_temporal_heads, behave_channel_heads, behave_len, dropout_enc, imu_type)
        self.imu_type = imu_type
        if imu_type != 'none':
            self.imu_transformer = Transformer(num_layer, imu_feature_dim, gre_k, imu_temporal_heads, imu_channel_heads, imu_len, dropout_enc, imu_type)

        # Example: output of behave_transformer is (64, 50, 8) for (batch_size, seq_len, feature_dim) => this will be flatten to (batch_sie, seq_len*feature_dim)
        #  => Input of linear layer is (400, 200)
        self.linear_behave = nn.Sequential(
            nn.Linear(behave_feature_dim * behave_len, (behave_feature_dim * behave_len) // 2),
            nn.ReLU(),
            nn.Dropout(dropout_linear),
            nn.Linear((behave_feature_dim * behave_len) // 2, target_len),
            nn.ReLU()
        )
        if imu_type != 'none':
            self.linear_imu = nn.Sequential(
                nn.Linear(imu_feature_dim * imu_len, (imu_feature_dim * imu_len) // 2),
                nn.ReLU(),
                nn.Dropout(dropout_linear),
                nn.Linear((imu_feature_dim * imu_len) // 2, target_len),
                nn.ReLU()
            )
            self.linear_behave_imu = nn.Linear(target_len*2, target_len)
        
    def forward(self, inputs):
        if self.imu_type != 'none':
            behave_inputs, imu_inputs = inputs
        else:
            behave_inputs = inputs

        behave_out = self.linear_behave(torch.flatten(self.behave_transformer(behave_inputs), start_dim=1, end_dim=2))

        if self.imu_type != 'none':
            imu_out = self.linear_imu(torch.flatten(self.imu_transformer(imu_inputs), start_dim=1, end_dim=2))
            concat_out = torch.concat([behave_out, imu_out], dim=-1)
            return self.linear_behave_imu(concat_out)
        else:
            return behave_out