from torch import nn

from model.layer_norm import LayerNorm
from model.multi_head_attention import MultiHeadAttention
from model.point_wise_mlp import PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, s_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=s_mask)
        x = self.dropout1(x)
        # 2. add and norm
        x = self.norm1(x + _x)
        
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
      
        # 4. add and norm
        x = self.norm2(x + _x)
        
        return x
