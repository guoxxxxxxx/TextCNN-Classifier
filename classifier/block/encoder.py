from torch import nn

from classifier.block.multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim=64, dropout=0.1, is_casual=False, max_seq_len=2048):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads, dropout, is_casual, max_seq_len)
        self.fnn_norm = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        norm_x = self.attn_norm(x)
        h = x + self.multi_head_attention(norm_x, norm_x, norm_x)
        out = h + self.feed_forward(self.fnn_norm(h))
        return out


class Encoder(nn.Module):
    def __init__(self, n, embed_dim, num_heads, hidden_dim=64, dropout=0.1, is_casual=False, max_seq_len=2048):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, hidden_dim, dropout, is_casual, max_seq_len) for _ in range(n)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)