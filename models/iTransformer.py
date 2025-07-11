import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, args):
        super(iTransformer, self).__init__()
        self.seq_len = args.seq_len
        self.output_attention = args.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(args.seq_len, args.d_model_for_itransformer, args.embed, args.freq,
                                                    args.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout,
                                      output_attention=args.output_attention), args.d_model_for_itransformer, args.n_heads),
                    args.d_model_for_itransformer,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation
                ) for l in range(args.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model_for_itransformer)
        )
        # Decoder
        self.projection = nn.Linear(args.d_model_for_itransformer, args.seq_len, bias=True)


    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        # print('enc_out.shape:',enc_out.shape)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # print('encoder.shape:',enc_out.shape)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # print('dec_out.shape:',dec_out.shape)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def forward(self, x_enc):
        dec_out = self.anomaly_detection(x_enc)
        return dec_out  # [B, L, D]
