from torch import nn
import torch as th
import math
from torchinfo import summary
import torch.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = th.zeros(max_len, dim_model)
        positions_list = th.arange(0, max_len, dtype=th.float).view(
            -1, 1
        )  # 0, 1, 2, 3, 4, 5
        division_term = th.exp(
            th.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )  # 1000^(2i/dim_model)

        pos_encoding[:, 0::2] = th.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = th.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: th.tensor) -> th.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding)


class TransformerDecisionNet(nn.Module):
    def __init__(
        self,
        vector_size: int = 128,
        hidden_dimension: int = 64,
        transformer_layers: int = 6,
        transformer_heads: int = 8,
    ):

        super().__init__()

        self._transformer_layers = transformer_layers
        self._transformer_heads = transformer_heads
        self.pe = PositionalEncoding(vector_size, 0.0, 10)
        self.mha = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    vector_size,
                    self._transformer_heads,
                    dim_feedforward=vector_size * 4,
                    batch_first=True,
                    activation=nn.SiLU()
                )
                for _ in range(self._transformer_layers)
            ]
        )
        self.SwiGLU = SwiGLU(vector_size*10, hidden_dimension, 16)

    def forward(self, x):
        x = self.pe(x)

        for attn in self.mha:
            x = attn(x)
        action = self.SwiGLU(th.flatten(x, start_dim=1))
        return action
    
class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # First branch with SiLU activation
        gate = self.silu(self.fc1(x))
        # Second branch without activation
        linear = self.fc2(x)
        # Element-wise product
        return self.out(gate * linear)
model = TransformerDecisionNet(128, 128, 6, 4)

print(summary(model, input_size=[4, 10, 128]))