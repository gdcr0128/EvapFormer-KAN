import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kan import KAN


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.kan_ffn = KAN(widths=[d_model, d_model * 2, d_model])

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        input_norm = torch.norm(x, p=2, dim=-1).detach().cpu().numpy()

        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        y = self.dropout(self.kan_ffn(x))
        return self.norm2(x + y), attn, input_norm


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        attention_counter = 0

        if not os.path.exists('attention_weights'):
            os.makedirs('attention_weights')

        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn, input_norm = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)

                attention_counter += 1
                np.savez(os.path.join('attention_weights', f'attention_layer_{attention_counter}.npz'),
                         attn=attn.cpu().numpy())
                np.savez(os.path.join('attention_weights', f'input_norm_layer_{attention_counter}.npz'),
                         input_norm=input_norm)

            x, attn, input_norm = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)

        else:
            for attn_layer in self.attn_layers:
                x, attn, input_norm = attn_layer(x, attn_mask, tau, delta)
                attns.append(attn)

                attention_counter += 1
                np.savez(os.path.join('attention_weights', f'attention_layer_{attention_counter}.npz'),
                         attn=attn.detach().cpu().numpy())
                np.savez(os.path.join('attention_weights', f'input_norm_layer_{attention_counter}.npz'),
                         input_norm=input_norm)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.kan_ffn = KAN(widths=[d_model, d_model * 2, d_model])

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, tau=tau)[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0])
        x = self.norm2(x)

        y = self.dropout(self.kan_ffn(x))
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None, d_model=None, output_size=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

        if projection is None and d_model is not None and output_size is not None:
            self.projection = KAN(widths=[d_model, d_model * 2, output_size])
        else:
            self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
