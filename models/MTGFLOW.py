# %%
from cgitb import reset
from turtle import forward, shape
from numpy import percentile
import torch.nn as nn
import torch.nn.functional as F
from models.NF import MAF
import torch
import numpy as np
import scipy
from models.fusion_gat import MTAD_GAT
# from models.iTransformer import iTransformer
    

def interpolate(tensor, index, target_size, mode='nearest', dim=0):
    print(tensor.shape)
    source_length = tensor.shape[dim]
    if source_length > target_size:
        raise AttributeError('no need to interpolate')
    if dim == -1:
        new_tensor = torch.zeros((*tensor.shape[:-1], target_size), dtype=tensor.dtype, device=tensor.device)
    if dim == 0:
        new_tensor = torch.zeros((target_size, *tensor.shape[1:],), dtype=tensor.dtype, device=tensor.device)
    scale = target_size // source_length
    reset = target_size % source_length
    # if mode == 'nearest':
    new_index = index
    new_tensor[new_index, :] = tensor
    new_tensor[:new_index[0], :] = tensor[0, :].unsqueeze(0)
    for i in range(source_length - 1):
        new_tensor[new_index[i]:new_index[i + 1], :] = tensor[i, :].unsqueeze(0)
    new_tensor[new_index[i + 1]:, :] = tensor[i + 1, :].unsqueeze(0)
    return new_tensor

class GNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GNN, self).__init__()
        
        # 多尺度卷积层
        self.conv_scales = nn.ModuleList([
            nn.Conv1d(input_size, hidden_size, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_spatial = nn.Linear(hidden_size * 3, hidden_size)  # 合并多尺度特征
        
        # 添加自适应权重
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, h, A):
        N, K, L, D = h.shape
        
        # 多尺度时间特征提取
        h_scales = []
        h_reshaped = h.transpose(2, 3).reshape(N*K, D, L)
        
        for i, conv in enumerate(self.conv_scales):
            h_scale = conv(h_reshaped)
            h_scale = h_scale.reshape(N, K, D, L).transpose(2, 3)
            h_scales.append(h_scale)
        
        # 自适应加权组合
        weights = self.softmax(self.scale_weights)
        h_multi = sum(w * h_s for w, h_s in zip(weights, h_scales))
        
        # 空间图卷积
        h_spatial = self.lin_n(torch.einsum('nkld,nkj->njld', h_multi, A))
        
        return h_spatial

import math
import torch.nn as nn
import matplotlib.pyplot as plt


def plot_attention(data, i, X_label=None, Y_label=None):
    '''
      Plot the attention model heatmap
      Args:
        data: attn_matrix with shape [ty, tx], cutted before 'PAD'
        X_label: list of size tx, encoder tags
        Y_label: list of size ty, decoder tags
    '''
    fig, ax = plt.subplots(figsize=(20, 8))  # set figure size
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)
    fig.colorbar(heatmap)
    # Set axis labels
    if X_label != None and Y_label != None:
        X_label = [x_label for x_label in X_label]
        Y_label = [y_label for y_label in Y_label]

        xticks = range(0, len(X_label))
        ax.set_xticks(xticks, minor=False)  # major ticks
        ax.set_xticklabels(X_label, minor=False, rotation=45)  # labels should be 'unicode'

        yticks = range(0, len(Y_label))
        ax.set_yticks(yticks, minor=False)
        ax.set_yticklabels(Y_label[::-1], minor=False)  # labels should be 'unicode'

        ax.grid(True)
        plt.show()
        plt.savefig('graph/attention{:04d}.jpg'.format(i))


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, c):
        super(ScaleDotProductAttention, self).__init__()
        self.w_q = nn.Linear(c, c)
        self.w_k = nn.Linear(c, c)
        self.w_v = nn.Linear(c, c)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        # swat_0.2

    def forward(self, x, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        shape = x.shape
        x_shape = x.reshape((shape[0], shape[1], -1))
        batch_size, length, c = x_shape.size()
        q = self.w_q(x_shape)
        k = self.w_k(x_shape)
        k_t = k.view(batch_size, c, length)  # transpose
        score = (q @ k_t) / math.sqrt(c)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        # 3. pass them softmax to make [0, 1] range
        score = self.dropout(self.softmax(score))

        return score, k


class MTGFLOW(nn.Module):

    def __init__(self, device, args, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout=0.1, model="MAF",
                 batch_norm=True):
        super(MTGFLOW, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=dropout)
        self.conv_layer = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        self.itransformer = iTransformer(args)
        if model == "MAF":
            # self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh', mode = 'zero')
            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size,
                          batch_norm=batch_norm, activation='tanh')

        self.attention = ScaleDotProductAttention(window_size * input_size)
        self.MGAT = MTAD_GAT(
        device,
        input_size,
        args.n_features,
        window_size,args,
        kernel_size=args.kernel_size,
        
        use_gatv2=args.use_gatv2,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        dropout=dropout,
        alpha=args.alpha)

    def forward(self, x, ):
        return self.test(x, ).mean()

    def test(self, x, ):
        # x: N X K X L X D
        if x.shape[3]!=1:
            x = x.permute(0,3,1,2)
            x = self.conv_layer(x)
            #x = x.squeeze(3)
            x = x.permute(0,2,3,1)
        full_shape = x.shape

        graph_structure=self.MGAT(x) #[batch_size, seq_len, n_features]
        graph_structure=graph_structure.permute(0,2,1)
        graph,_ = self.attention(graph_structure)    
        # graph, _ = self.attention(x)
        self.graph = graph
        # reshape: N*K, L, D
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        h = self.itransformer(x)
        h, _ = self.rnn(h)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        h = self.gcn(h, graph)

        # reshappe N*K*L,H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, full_shape[3]))
        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2], h).reshape([full_shape[0], -1])  
        log_prob = log_prob.mean(dim=1)
        graph_enhanced_log_prob = self.enhance_log_prob_with_graph(log_prob, graph)

        return graph_enhanced_log_prob

    def get_graph(self):
        return self.graph

    def locate(self, x, ):
        # x: N X K X L X D
        if x.shape[3]!=1:
            x = x.permute(0,3,1,2)
            x = self.conv_layer(x)
            #x = x.squeeze(3)
            x = x.permute(0,2,3,1)
        full_shape = x.shape

        graph_structure=self.MGAT(x) #[batch_size, seq_len, n_features]
        graph_structure=graph_structure.permute(0,2,1)
        graph,_ = self.attention(graph_structure) 
        # reshape: N*K, L, D
        self.graph = graph
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h = self.itransformer(x)
        h, _ = self.rnn(h)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        h = self.gcn(h, graph)

        # reshappe N*K*L,H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, full_shape[3]))
        a = self.nf.log_prob(x, full_shape[1], full_shape[2], h)
        log_prob, z = a[0].reshape([full_shape[0], full_shape[1], -1]), a[1].reshape([full_shape[0], full_shape[1], -1])

        return log_prob.mean(dim=2), z.reshape((full_shape[0] * full_shape[1], -1))


class MTGFLOWZL(nn.Module):

    def __init__(self, device, args, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout=0.1, model="MAF",
                 batch_norm=True):
        super(MTGFLOWZL, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=dropout)
        self.conv_layer = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)
        # self.itransformer = iTransformer(args)
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        # 新增：GRU层
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if 2 > 1 else 0
        )
        
        # GRU输出转换层（因为是双向GRU，输出维度是hidden_size*2）
        self.gru_transform = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        if model == "MAF":
            # self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh', mode = 'zero')
            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size,
                          batch_norm=batch_norm, activation='tanh')

        self.gcn_mlp = nn.Sequential(
            nn.Linear(n_sensor * window_size, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, n_sensor * window_size * 32),
        )

        self.ann = nn.Sequential(
            nn.Linear(n_sensor * window_size * 32, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
        )
        self.attention = ScaleDotProductAttention(window_size * input_size)
        self.MGAT = MTAD_GAT(
            device,
            input_size,
            args.n_features,
            window_size,args,
            kernel_size=args.kernel_size,
            
            use_gatv2=args.use_gatv2,
            feat_gat_embed_dim=args.feat_gat_embed_dim,
            time_gat_embed_dim=args.time_gat_embed_dim,
            dropout=dropout,
            alpha=args.alpha)

    def forward(self, x, ):
        hid, log_prob, h_gcn = self.test(x, )
        return hid, log_prob.mean(), h_gcn, log_prob

    def test(self, x, ):
        # x: N X K X L X D 
        # import pdb; pdb.set_trace()
        if x.shape[3]!=1:
            x = x.permute(0,3,1,2)
            x = self.conv_layer(x)
            #x = x.squeeze(3)
            x = x.permute(0,2,3,1) 
        full_shape = x.shape #torch.Size([128, 25, 60, 1])
        # print('full_shape.shape:',x.shape)

        graph_structure=self.MGAT(x) #[batch_size, seq_len, n_features]
        # print('MGAT.shape:',graph_structure.shape)

        graph_structure=graph_structure.permute(0,2,1)
        graph,_ = self.attention(graph_structure)    
        # graph, _ = self.attention(x)
        self.graph = graph
        # print('attention.shape:',graph.shape)
        # reshape: N*K, L, D
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) #torch.Size([3200, 60, 1])
        h, _ = self.rnn(x) #torch.Size([6400, 60, 32])
        # h = self.temporal_encoder(x)  # TSLANet-based temporal encoder
        
        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2])) #torch.Size([256, 25, 60, 32])
        
        h_gcn = self.gcn(h, graph)
        # print('gcn.shape:',h_gcn.shape)
        # import pdb; pdb.set_trace()

        batch_size, n_nodes, seq_len, hidden_dim = h_gcn.shape
        gru_in = h_gcn.transpose(1, 2).reshape(batch_size * seq_len, n_nodes, hidden_dim)
        gru_out, _ = self.gru(gru_in)
        gru_out = self.gru_transform(gru_out)
        h_gcn = gru_out.reshape(batch_size, seq_len, n_nodes, hidden_dim).transpose(1, 2) #torch.Size([256, 25, 60, 32]) batch_size, n_sensor, window_size, hidden_size 
        
        shape_hgcn = h_gcn.shape
        h_gcn = self.gcn_mlp(x.reshape((shape_hgcn[0], -1))).reshape(*shape_hgcn) + h_gcn
        h_gcn_for_ann = h_gcn.reshape(full_shape[0], -1)

        # reshappe N*K*L,H
        h_gcn_reshape = h_gcn.reshape((-1, h_gcn.shape[3]))
        x = x.reshape((-1, full_shape[3]))

        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2], h_gcn_reshape).reshape([full_shape[0], -1]) 
        log_prob = log_prob.mean(dim=1)
        graph_enhanced_log_prob = self.enhance_log_prob_with_graph(log_prob, graph)
        
        return self.ann(h_gcn_for_ann), graph_enhanced_log_prob, h_gcn

    def get_graph(self):
        return self.graph

    def _DistanceSquared(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-6)

        return dist

    def _CalGamma(self, v):
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b

        return out

    def _TwowaydivergenceLoss(self, P_, Q_, select=None):
        EPS = 1e-5
        losssum1 = P_ * torch.log(Q_ + EPS)
        losssum2 = (1 - P_) * torch.log(1 - Q_ + EPS)
        losssum = -1 * (losssum1 + losssum2)
        return losssum.mean()

    def _Similarity(self, dist, gamma, v=100, h=1, pow=2):
        dist_rho = dist

        dist_rho[dist_rho < 0] = 0
        Pij = (
                gamma
                * torch.tensor(2 * 3.14)
                * gamma
                * torch.pow((1 + dist_rho / v), exponent=-1 * (v + 1))
        )
        return Pij

    def enhance_log_prob_with_graph(self, base_log_prob, graph):
        """
        Enhances log probability scores using graph structure insights:
        1. Similar graph structures for temporally adjacent normal samples
        2. Different graph structures for anomalous vs normal samples
        
        Args:
            base_log_prob: Base log probability from normalizing flow
            graph: Dynamic graph structure from MGAT
        
        Returns:
            Enhanced log probability scores
        """
        batch_size = graph.shape[0]
        
        # Skip if batch size is too small for meaningful comparison
        if batch_size <= 1:
            return base_log_prob
        
        graph_sim = torch.zeros_like(base_log_prob, device=base_log_prob.device)
        
        # Calculate graph similarity between adjacent samples
        for i in range(1, batch_size):
            # Frobenius norm of difference between adjacent graphs
            graph_diff = torch.norm(graph[i] - graph[i-1], p='fro') 
            
            # Normalize by graph size
            norm_factor = graph[i].shape[0] * graph[i].shape[1]
            graph_diff = graph_diff / torch.sqrt(torch.tensor(norm_factor, dtype=torch.float, device=graph_diff.device))
            
            # Convert to similarity (higher for similar graphs)
            graph_sim[i] = torch.exp(-graph_diff * 5.0)  # Scale factor 5.0 can be tuned
        
        # For first sample, use similarity with second sample
        if batch_size > 1:
            graph_sim[0] = graph_sim[1]
        
        # Assuming lower log probability indicates potential anomalies
        log_prob_normalized = (base_log_prob - base_log_prob.mean()) / (base_log_prob.std() + 1e-8)
        anomaly_scores = torch.sigmoid(-log_prob_normalized)  # Higher for potential anomalies
        
        # - For normal samples: maintain or slightly boost log probability
        # - For anomalous samples: further reduce log probability if graph structure differs
        
        # Temporal consistency factor (higher for samples with consistent graph structure)
        temporal_consistency = 1.0 + 0.2 * graph_sim
        
        # Anomaly amplification factor (reduces log probability for anomalies with distinct graph structure)
        anomaly_amplification = 1.0 - 0.5 * anomaly_scores * (1.0 - graph_sim)
        
        # Combine factors
        enhancement_factor = temporal_consistency * anomaly_amplification
        
        # Apply enhancement (multiply by factor to preserve sign of log probability)
        enhanced_log_prob = base_log_prob * enhancement_factor
    
        return enhanced_log_prob

    def LossManifold(
            self,
            input_data,
            latent_data,
            v_input,
            v_latent,
            w=1,
            metric="euclidean",
            label=None,
    ):
        # normalize the input and latent data
        # import pdb; pdb.set_trace()
        # input_data = input_data / torch.std(input_data, dim=0).detach()
        # latent_data = latent_data / torch.std(latent_data, dim=0).detach()
        # print("input_data:", input_data.shape)
        # print("latent_data:", latent_data.shape)

        batch_size = input_data.shape[0]

        data_1 = input_data[: input_data.shape[0] // 2]
        dis_P = self._DistanceSquared(data_1, data_1)
        # print("dis_P:", dis_P.shape)

        latent_data_1 = latent_data[: input_data.shape[0] // 2]

        dis_P_2 = dis_P  # + nndistance.reshape(1, -1)

        P_2 = self._Similarity(dist=dis_P_2, gamma=self._CalGamma(v_input), v=v_input)
        # print("P_2:", P_2.shape)

        latent_data_2 = latent_data[(input_data.shape[0] // 2):]
        dis_Q_2 = self._DistanceSquared(latent_data_1, latent_data_2)
        # print("dis_Q_2:", dis_Q_2.shape)

        Q_2 = self._Similarity(
            dist=dis_Q_2,
            gamma=self._CalGamma(v_latent),
            v=v_latent,
        )
        # print("Q_2:", Q_2.shape)

        eye_mask = torch.eye(P_2.shape[0]).to(input_data.device)
        loss_ce_posi = self._TwowaydivergenceLoss(
            P_=P_2[eye_mask == 1], Q_=Q_2[eye_mask == 1]
        )
        # P_2_copy = P_2.detach()
        # label_matrix = label.reshape(1, -1).repeat(label.shape[0],1)
        # lable_mask = (label_matrix-label_matrix.T)!=0
        # P_2_copy[lable_mask] = 0.0
        loss_ce_nega = self._TwowaydivergenceLoss(
            P_=P_2[eye_mask == 0], Q_=Q_2[eye_mask == 0]
        )
        w1, w2 = 1 / (1 + w), w / (1 + w)
        return w2 * loss_ce_nega, w1 * loss_ce_posi / batch_size

    # def EnhancedLossManifold(
    #     self,
    #     input_data,
    #     latent_data,
    #     graph_adj=None,  # Optional graph adjacency matrix
    #     v_input=100,
    #     v_latent=0.01,
    #     w=1.0,
    #     anomaly_factor=2.0,
    #     eps=1e-6,
    #     sample_rate=1.0,  # For efficiency with large batches
    #     metric="euclidean",
    #     label=None,
    # ):
    #     """
    #     Enhanced manifold alignment loss with graph structure awareness and anomaly sensitivity.
    #     """
    #     batch_size = input_data.shape[0]
    #     half_size = batch_size // 2
        
    #     # Split data for paired analysis
    #     data_1 = input_data[:half_size]
    #     latent_data_1 = latent_data[:half_size]
    #     latent_data_2 = latent_data[half_size:]
        
    #     # Sampling for efficiency (if needed)
    #     if sample_rate < 1.0 and half_size > 10:
    #         sample_size = max(int(half_size * sample_rate), 2)
    #         idx = torch.randperm(half_size, device=input_data.device)[:sample_size]
    #         data_1 = data_1[idx]
    #         latent_data_1 = latent_data_1[idx]
    #         latent_data_2 = latent_data_2[idx]
        
    #     # Compute pairwise distances
    #     dis_P = self._DistanceSquared(data_1, data_1)
    #     dis_P = torch.clamp(dis_P, min=eps)
        
    #     dis_Q = self._DistanceSquared(latent_data_1, latent_data_2)
    #     dis_Q = torch.clamp(dis_Q, min=eps)
        
    #     # Compute similarities
    #     gamma_input = self._CalGamma(v_input)
    #     gamma_latent = self._CalGamma(v_latent)
        
    #     P_sim = self._Similarity(dist=dis_P, gamma=gamma_input, v=v_input)
    #     Q_sim = self._Similarity(dist=dis_Q, gamma=gamma_latent, v=v_latent)
        
    #     # Identify potential anomalies
    #     row_var = torch.var(dis_P, dim=1)
    #     if torch.max(row_var) > eps:
    #         anomaly_weights = torch.sigmoid((row_var - torch.mean(row_var)) / (torch.std(row_var) + eps))
    #         enhanced_weights = 1.0 + (anomaly_factor - 1.0) * anomaly_weights
    #         enhanced_weights = enhanced_weights.unsqueeze(1)
    #     else:
    #         enhanced_weights = torch.ones(P_sim.shape[0], 1, device=P_sim.device)
        
    #     # Graph structure consistency - decoupled from direct comparison
    #     structure_loss = torch.tensor(0.0, device=input_data.device)
    #     if graph_adj is not None:
    #         if half_size > 1:
    #             # Calculate graph density and structural metrics
    #             graph_density = torch.mean(graph_adj[:half_size])
    #             graph_diag = torch.diagonal(graph_adj[:half_size], dim1=1, dim2=2)
    #             graph_trace = torch.mean(graph_diag)
                
    #             # Use graph metrics as regularization terms
    #             structure_loss = torch.abs(graph_density - graph_trace)
        
    #     # Calculate standard contrastive loss
    #     eye_mask = torch.eye(P_sim.shape[0], device=P_sim.device)
        
    #     # Apply anomaly-enhanced weights
    #     weighted_Q = Q_sim.clone()
    #     weighted_Q[eye_mask == 0] = weighted_Q[eye_mask == 0] * enhanced_weights.repeat(1, P_sim.shape[0])[eye_mask == 0]
        
    #     # Calculate divergence losses
    #     loss_ce_posi = self._TwowaydivergenceLoss(
    #         P_=P_sim[eye_mask == 1], 
    #         Q_=weighted_Q[eye_mask == 1]
    #     )
        
    #     loss_ce_nega = self._TwowaydivergenceLoss(
    #         P_=P_sim[eye_mask == 0], 
    #         Q_=weighted_Q[eye_mask == 0]
    #     )
        
    #     # Dynamic weighting
    #     if torch.mean(anomaly_weights) > 0.5:
    #         w = w * 1.2
        
    #     w1, w2 = 1 / (1 + w), w / (1 + w)
        
    #     return w2 * loss_ce_nega, w1 * loss_ce_posi / batch_size, structure_loss

class test(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout=0.1, model="MAF",
                 batch_norm=True):
        super(test, self).__init__()

        if model == "MAF":
            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, batch_norm=batch_norm,
                          activation='tanh', mode='zero')
        self.attention = ScaleDotProductAttention(window_size * input_size)

    def forward(self, x, ):
        return self.test(x, ).mean()

    def test(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        full_shape = x.shape
        x = x.reshape((full_shape[0] * full_shape[1], full_shape[2], full_shape[3]))
        x = x.reshape((-1, full_shape[3]))
        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2]).reshape(
            [full_shape[0], full_shape[1], -1])  # *full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=1)
        return log_prob

    def locate(self, x, ):
        # x: N X K X L X D 
        x = x.unsqueeze(2).unsqueeze(3)
        full_shape = x.shape

        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # reshappe N*K*L,H
        x = x.reshape((-1, full_shape[3]))
        a = self.nf.log_prob(x, full_shape[1], full_shape[2])  # *full_shape[1]*full_shape[2]
        log_prob, z = a[0].reshape([full_shape[0], full_shape[1], -1]), a[1].reshape([full_shape[0], full_shape[1], -1])

        return log_prob.mean(dim=2), z.reshape((full_shape[0] * full_shape[1], -1))
