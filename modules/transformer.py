from torch import nn
import torch
import numpy as np
from .attention import (
    MultiHeadAttention,
    QKVCreator
)


class Transformer(nn.Module):
    """Reproduction of paper `Attention Is All You Need`."""
    def __init__(self, n_encoder=6, n_decoder=6, d_model=512, d_output=512):
        super().__init__()
        self.encoder = nn.Sequential(*[TransformerEncoderUnit(d_model=d_model) for i in range(n_encoder)])
        self.decoder = nn.ModuleList([TransformerDecoderUnit(d_model=d_model) for i in range(n_decoder)])
        self.linear = nn.Linear(d_model, d_output)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, embeddings):
        memory = self.encoder(embeddings)
        for layer in self.decoder:
            embeddings = layer(embeddings, memory)
        output = self.linear(embeddings)
        output = self.softmax(output)
        return output



class PositionalEncoder(nn.Module):
    def __init__(self, seq_length, d_model, number_start=0):
        """
        Compute the positional encoding described in paper "Attention Is All You Need".

        :param seq_length:      sequence length
        :param d_moddel:        dimension of Transformer Encoder
        :param number_start:    0 or 1, for machine-like or human-like
        """
        super().__init__()
        self.positional_encoding = self.compute_positional_encoding(seq_length, d_model, number_start)
        return self.positional_encoding

    def compute_positional_encoding(self, s, d, number_start=0):
        """

        :param s:   sequence length
        :param d:   d_model
        :return:
        """
        if number_start==0:
            positional_encoding = self._compute_positional_encoding_machine_like(s, d)

        return positional_encoding

    def _compute_positional_encoding_machine_like(self, s, d):
        """
        Refer to GLM(General Language Model)

        :param s:
        :param d:
        :return:
        """
        single_line_position = np.arange(d)
        single_line_position[1::2] = single_line_position[1::2] - 1
        single_line_position_mapped = 10000**(single_line_position/d)
        single_column_position = np.arange(s)
        single_line_position_mapped = single_line_position_mapped.reshape(1,-1)
        single_column_position = single_column_position.reshape(-1,1)
        positional_encoding = single_column_position/single_line_position_mapped

        positional_encoding = torch.tensor(positional_encoding, requires_grad=False)
        positional_encoding[::2] = torch.sin(positional_encoding[::2])
        positional_encoding[1::2] = torch.cos(positional_encoding[1::2])
        return positional_encoding



class TransformerEncoderUnit(nn.Module):
    def __init__(self, d_model, multi_head=8, d_ff=2048):
        super().__init__()
        self.qkv_creator = QKVCreator(d_model)
        self.multi_head_attention = MultiHeadAttention(d_model, multi_head)
        self.layer_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)

    def forward(self, embeddings):
        Q, K, V = self.qkv_creator(embeddings)
        output = self.multi_head_attention(Q, K, V)

        output = embeddings + output
        output = self.layer_norm(output)

        output_ffn = self.ffn(output)
        output = output + output_ffn
        output = self.layer_norm(output)
        return output

class TransformerDecoderUnit(nn.Module):
    def __init__(self, d_model, multi_head=8, d_ff=2048):
        super().__init__()
        self.qkv_creator = QKVCreator(d_model)
        self.masked_multi_head_attention = MultiHeadAttention(d_model, multi_head, mask=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.q_creator = QKVCreator(d_model,q_only=True)
        self.kv_creator = QKVCreator(d_model, kv_only=True)
        self.multi_head_attention = MultiHeadAttention(d_model, multi_head)
        self.ffn = FeedForwardNetwork(d_model, d_ff)

    def forward(self, embeddings, memory=None):#, encoder_keys=None, encoder_values=None):
        Q, K, V = self.qkv_creator(embeddings)
        output = self.masked_multi_head_attention(Q, K, V)
        output += embeddings
        output = self.layer_norm(output)
        decoder_q = self.q_creator(output)
        memory_keys, memory_values = self.kv_creator(memory)

        # output from cross attention
        # output_x_att = self.multi_head_attention(decoder_q, encoder_keys, encoder_values)
        output_x_att = self.multi_head_attention(decoder_q, memory_keys, memory_values)

        output += output_x_att
        output = self.layer_norm(output)
        output_ffn = self.ffn(output)
        output += output_ffn
        output = self.layer_norm(output)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=512, d_hidden=2048, activate=nn.ReLU):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_hidden)
        self.activate = activate()
        self.layer2 = nn.Linear(d_hidden, d_model)

    def forward(self, embeddings):
        output = self.layer1(embeddings)
        output = self.activate(output)
        output = self.layer2(output)
        return output


if __name__ == "__main__":
    from loguru import logger
    pe = PositionalEncoder(512, 1000)
    print(pe.positional_encoding.shape)

    from matplotlib import pyplot as plt
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    data = pe.positional_encoding.cpu().numpy()
    for i in range(5):
        plt.plot(data[i*2,:])
    plt.show()





