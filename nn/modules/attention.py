from torch import nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_input, d_k=None, d_v=None, mask=None):
        """

        :param d_input:     is the same as d_model in Attention Is All You Need
        :param d_k:
        :param d_v:
        :param mask:
        """
        super().__init__()
        if d_k is None: d_k = d_input
        if d_v is None: d_v = d_input
        self.W_q = Parameter(torch.randn((d_input, d_k)))
        self.W_k = Parameter(torch.randn((d_input, d_k)))
        self.W_v = Parameter(torch.randn((d_input, d_v)))

        self.scaled_dpa = ScaledDotProductAttention(d_k, mask=mask)
        pass

    def forward(self, embeddings):
        """

        :param embeddings:  embedding matrix
        :return:
        """
        Q = torch.matmul(embeddings, self.W_q)
        K = torch.matmul(embeddings, self.W_k)
        V = torch.matmul(embeddings, self.W_v)
        output = self.scaled_dpa(Q, K, V)
        return output


def get_mask(d_mask, diagnol, dtype=torch.float32):
    mask = -torch.inf * torch.ones((d_mask, d_mask), requires_grad=False, dtype=dtype)
    mask = torch.triu(mask, diagnol)
    return mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, mask=False, d_mask=None, diagnol=1):
        super().__init__()
        self.d_k = d_k

        self.mask = mask
        self.diagnol = diagnol

    def forward(self, Q, K, V):
        sim = torch.matmul(Q, torch.permute(K, (0, 2, 1)))
        sim = sim / self.d_k ** 0.5
        if self.mask:
            mask = get_mask(Q.shape[1], self.diagnol, Q.dtype)
            sim += mask

        weights = F.softmax(sim, dim=-1)
        output = torch.matmul(weights, V)
        return output


class QKVCreator(nn.Module):
    def __init__(self, d_input, d_k=None, d_v=None, q_only=False, kv_only=False):
        """

        :param d_input:     is the same as d_model in Attention Is All You Need
        :param d_k:
        :param d_v:
        :param mask:
        """
        super().__init__()
        if d_k is None: d_k = d_input
        if d_v is None: d_v = d_input
        W_q = Parameter(torch.randn((d_input, d_k)))
        W_k = Parameter(torch.randn((d_input, d_k)))
        W_v = Parameter(torch.randn((d_input, d_v)))
        if q_only:
            self.Ws = nn.ParameterList([W_q])
            del W_k, W_v
        elif kv_only:
            self.Ws = nn.ParameterList([W_k, W_v])
            del W_q
        else:
            self.Ws = nn.ParameterList([W_q, W_k, W_v])

        self.q_only = q_only
        self.kv_only = kv_only

    def forward(self, embeddings):
        values = [torch.matmul(embeddings, W) for W in self.Ws]
        if len(values) == 1:
            values = values[0]
        return values


class MultiHeadAttention(nn.Module):
    def __init__(self, d_input, h=8, d_k=None, d_v=None, mask=None):
        super().__init__()
        if d_k is None:
            d_k = d_input // h
        if d_v is None:
            d_v = d_input // h

        self.d_k = d_k
        self.h = h

        self.scaled_dpa = ScaledDotProductAttention(d_k, mask=mask, d_mask=d_k)

        self.Q_rotation_matrices = nn.ParameterList([Parameter(torch.randn((d_input, d_k)))
                                                     for i in range(h)])
        self.K_rotation_matrices = nn.ParameterList([Parameter(torch.randn((d_input, d_k)))
                                                     for i in range(h)])
        self.V_rotation_matrices = nn.ParameterList([Parameter(torch.randn((d_input, d_v)))
                                                     for i in range(h)])

        self.linear = nn.Linear(d_v * h, d_input, bias=False)

    def forward(self, Q, K, V):

        Qs = [torch.matmul(Q, W_i) for W_i in self.Q_rotation_matrices]
        Ks = [torch.matmul(K, W_i) for W_i in self.K_rotation_matrices]
        Vs = [torch.matmul(V, W_i) for W_i in self.V_rotation_matrices]
        outputs = [self.scaled_dpa(Qs[i], Ks[i], Vs[i]) for i in range(self.h)]
        output = torch.concat(outputs, dim=-1)
        output = self.linear(output)
        return output
