import torch
import torch.nn as nn


class TransformerLM(nn.Module):
    """Transformer Language Model"""
    def __init__(self, config, vocab):
        super(TransformerLM, self).__init__()

        num_layers = config.get('num_layers', 6)

        self.input_layer = TransformerInputLayer(config, vocab)
        self.layers = [TransformerLayer(config) for _ in range(num_layers)]
        self.output_layer = TransformerOutputLayer(config, vocab)

        # Weight tying. TODO: check if it works
        self.output_layer.project.weight = self.input_layer.embed.weight

    def forward(self, x):
        h = self.input_layer(x)

        for layer in self.layers:
            h = layer(h)

        out = self.out_layer(h)

        return out


class TransformerInputLayer(nn.Module):
    def __init__(self, config, vocab):
        super(TransformerInputLayer, self).__init__()

        embed_dim = config.get('embed_dim', 512)
        max_seq_len = config.get('max_seq_len', 512)

        self.embed = nn.Embedding(len(vocab), embed_dim, padding_idx=vocab['<pad>'])
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x):
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.embed(x)
        h = h + self.pos_embed(positions).expand_as(h)
        h = self.dropout(h)

        return h


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super(TransformerLayer, self).__init__()

        embed_dim = config.get('embed_dim', 512)
        dropout_p = config.get('dropout_p', 0.1)
        num_heads = config.get('num_heads', 8)
        hidden_dim = config.get('hidden_dim', 2048)

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_p)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.ln_1 = nn.LayerNorm(embed_dim, eps=1e-12)
        self.ln_2 = nn.LayerNorm(embed_dim, eps=1e-12)

        self.apply(self.init_weights)

    # def init_weights(self, module):
    #     if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #     if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
    #         module.bias.data.zero_()

    def forward(self, x):
        attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=x.device, dtype=x.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        # Attention
        x = self.ln_1(x)
        x_res, _ = self.attention(x, x, x, attn_mask=attn_mask, need_weights=False)
        x_res = self.dropout(x_res)
        x = x_res + x

        # Feed-forward network
        x = self.ln_2(x)
        x_res = self.ffn(x)
        x_res = self.dropout(x_res)
        x = x_res + x

        return x


class TransformerOutputLayer(nn.Module):
    def __init__(self, config, vocab):
        super(TransformerOutputLayer, self).__init__()

        self.project = nn.Linear(config.embed_dim, len(vocab), bias=False)

    def forward(self, x):
        return self.project(x)
