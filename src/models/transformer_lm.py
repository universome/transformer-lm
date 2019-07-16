from typing import List

import torch
import torch.nn as nn


class TransformerLM(nn.Module):
    """Transformer Language Model"""
    def __init__(self, config, vocab):
        super(TransformerLM, self).__init__()

        num_layers = config.get('num_layers', 6)

        self.input_layer = TransformerInputLayer(config, vocab)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(num_layers)])
        self.output_layer = TransformerOutputLayer(config, vocab)

        # Weight tying. TODO: check if it works
        self.output_layer.project.weight = self.input_layer.embed.weight

    def forward(self, x, return_state:bool=False, cache:List[torch.Tensor]=None):
        state = [self.input_layer(x)]

        for layer in self.layers:
            state.append(layer(state[-1]))

        logits = self.output_layer(state[-1])

        if return_state:
            return logits, torch.stack(state)
        else:
            return logits


    def cached_forward(self, x, cache:List[torch.Tensor]):
        """
        Performs forward pass with cached state
        Args:
            - x: token indices
            - cache: list of intermediate states of size 1 + len(self.layers)
        """
        state = [self.input_layer.cached_forward(x, cache[0])]

        for layer, layer_cache in zip(self.layers, cache[1:]):
            state.append(layer.cached_forward(state[-1], layer_cache))

        logits_last = self.output_layer(state[-1][:, -1:])

        return logits_last, torch.stack(state)


class TransformerInputLayer(nn.Module):
    def __init__(self, config, vocab):
        super(TransformerInputLayer, self).__init__()

        d_model = config.get('d_model', 512)
        max_seq_len = config.get('max_seq_len', 512)

        self.embed = nn.Embedding(len(vocab), d_model, padding_idx=vocab.stoi['<pad>'])
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(config.get('dropout_p', 0.1))

    def forward(self, x):
        h = self.embed(x)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = h + self.pos_embed(positions).expand_as(h)
        h = self.dropout(h)

        return h

    def cached_forward(self, x, cache:torch.Tensor=None):
        x_last = x[:, -1:].long()
        pos_last = torch.full((x.size(0), 1), x.size(1) - 1, device=x.device, dtype=x_last.dtype)
        h_last = self.embed(x_last) + self.pos_embed(pos_last)

        return torch.cat([cache, h_last], dim=1)


class TransformerLayer(nn.Module):
    """TODO: forward and cached_forward have too much in common"""
    def __init__(self, config):
        super(TransformerLayer, self).__init__()

        d_model = config.get('d_model', 512)
        dropout_p = config.get('dropout_p', 0.1)
        num_heads = config.get('num_heads', 8)
        ffn_hid_dim = config.get('ffn_hid_dim', 2048)

        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_p)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hid_dim),
            nn.ReLU(),
            nn.Linear(ffn_hid_dim, d_model)
        )
        self.ln_1 = nn.LayerNorm(d_model, eps=1e-12)
        self.ln_2 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(config.get('dropout_p', 0.1))

    def forward(self, x):
        attn_mask = torch.full((x.size(1), x.size(1)), -float('Inf'), device=x.device, dtype=x.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        # Attention
        x = self.ln_1(x)

        # Applying attention
        x_t = x.transpose(0, 1)
        x_res, _ = self.attention(x_t, x_t, x_t, attn_mask=attn_mask, need_weights=False)
        x_res = x_res.transpose(0, 1)

        x_res = self.dropout(x_res)
        x = x_res + x

        # Feed-forward network
        x = self.ln_2(x)
        x_res = self.ffn(x)
        x_res = self.dropout(x_res)
        x = x_res + x

        return x

    def cached_forward(self, x:torch.Tensor, cache:torch.Tensor):
        attn_mask = torch.zeros(1, x.size(1), device=x.device, dtype=x.dtype)

        # Attention
        x = self.ln_1(x)

        # Applying attention
        x_t = x.transpose(0, 1)
        x_res, _ = self.attention(x_t[-1:], x_t, x_t, attn_mask=attn_mask, need_weights=False)
        x_res = x_res.transpose(0, 1)

        x_res = self.dropout(x_res)
        x = x_res + x

        # Feed-forward network
        x = self.ln_2(x)
        x_res = self.ffn(x)
        x_res = self.dropout(x_res)
        x = x_res + x

        return torch.cat([cache, x[:, -1:]], dim=1)


class TransformerOutputLayer(nn.Module):
    def __init__(self, config, vocab):
        super(TransformerOutputLayer, self).__init__()

        self.project = nn.Linear(config.d_model, len(vocab), bias=False)

    def forward(self, x):
        return self.project(x)
