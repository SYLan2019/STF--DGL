import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.Mamba import Mamba

from models.modules import (
    ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    Forecasting_Model,
    ReconstructionModel,
)

class SimpleCNN(nn.Module):
    def __init__(self,input_size,output_size):
        super(SimpleCNN, self).__init__()
        self.conv1x1 = nn.Conv1d(input_size, output_size, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1x1(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, emb_dim, att_dropout=0.0):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5

        # Initialize linear transformations for Q, K, V
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        # Output projection layer
        self.proj_out = nn.Linear(emb_dim, emb_dim)

        # Dropout layer for attention weights
        self.dropout = nn.Dropout(att_dropout)

    def forward(self, A, B):
        '''
        :param A: Tensor of shape [batch_size, seq_len, emb_dim] (e.g., [256, 100, 25])
        :param B: Tensor of shape [batch_size, seq_len, emb_dim] (e.g., [256, 100, 25])
        :return: Tensor of shape [batch_size, seq_len, emb_dim]
        '''

        Q = self.Wq(A)  # [256, 100, 25]
        K = self.Wk(B)  # [256, 100, 25]
        V = self.Wv(B)  # [256, 100, 25]

        # Compute attention scores
        att_scores = torch.einsum('bij,bkj->bik', Q, K)  # [256, 100, 100]
        att_scores = att_scores * self.scale
        att_scores = F.softmax(att_scores, dim=-1)
        att_scores = self.dropout(att_scores)

        # Apply attention to V
        att_output = torch.einsum('bij,bjk->bik', att_scores, V)  # [256, 100, 25]

        # Project output to original dimension
        output = self.proj_out(att_output)  # [256, 100, 25]

        return output
    
class FeatureFusionAttention(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureFusionAttention, self).__init__()
        self.feature_dim = feature_dim
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
    def forward(self, x, h_feat, h_temp):
        """
        Args:
            x: 原始特征 [batch_size, seq_len, feature_dim]
            h_feat: 空间特征 [batch_size, seq_len, feature_dim]
            h_temp: 时间特征 [batch_size, seq_len, feature_dim]
        Returns:
            fused_features: 融合后的特征 [batch_size, seq_len, feature_dim]
            attention_weights: 三个特征的注意力权重 [batch_size, seq_len, 3]
        """
        # 计算每个特征的注意力分数
        x_score = self.attention(x)        # [batch_size, seq_len, 1]
        feat_score = self.attention(h_feat) # [batch_size, seq_len, 1]
        temp_score = self.attention(h_temp) # [batch_size, seq_len, 1]
        
        # 将三个分数拼接
        scores = torch.cat([x_score, feat_score, temp_score], dim=-1)  # [batch_size, seq_len, 3]
        
        # 通过softmax获得注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, 3]
        
        # 使用注意力权重融合特征
        fused_features = (x * attention_weights[:, :, 0:1] + 
                         h_feat * attention_weights[:, :, 1:2] + 
                         h_temp * attention_weights[:, :, 2:3])  # [batch_size, seq_len, feature_dim]
        
        return fused_features, attention_weights


class OptimizedFeatureFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=1, dropout=0):
        super(OptimizedFeatureFusion, self).__init__()
        self.feature_dim = feature_dim
        
        # 1. 简化的多头交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 2. 特征门控单元
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        # 3. 特征增强模块
        self.enhancement = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 4. 自适应权重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(feature_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3), 
            nn.Softmax(dim=-1)
        )
        
        # 5. 残差连接
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x, h_feat, h_temp):
        """
        Args:
            x: 原始特征 [batch_size, seq_len, feature_dim]
            h_feat: 空间特征 [batch_size, seq_len, feature_dim]
            h_temp: 时间特征 [batch_size, seq_len, feature_dim]
        Returns:
            fused_features: 融合后的特征 [batch_size, seq_len, feature_dim]
            attention_weights: 特征权重 [batch_size, seq_len, 3]
        """
        # 1. 交叉注意力增强
        h_feat_enhanced, _ = self.cross_attention(h_feat.transpose(0, 1), h_temp.transpose(0, 1), h_temp.transpose(0, 1))

        h_feat_enhanced = h_feat_enhanced.transpose(0, 1)
        
        # 2. 特征门控
        gate = self.gate(torch.cat([h_feat, h_feat_enhanced], dim=-1))
        h_feat_gated = h_feat * gate + h_feat_enhanced * (1 - gate)
        
        # 3. 特征拼接和增强
        concat_features = torch.cat([x, h_feat_gated, h_temp], dim=-1)
        enhanced_features = self.enhancement(concat_features)
        
        # 4. 自适应权重生成
        fusion_weights = self.weight_generator(concat_features)
        
        # 5. 加权融合
        weighted_features = (
            x * fusion_weights[:, :, 0:1] + 
            h_feat_gated * fusion_weights[:, :, 1:2] + 
            h_temp * fusion_weights[:, :, 2:3]
        )
        
        # 6. 残差连接和归一化
        fused_features = self.norm(weighted_features + enhanced_features)
        
        return fused_features, fusion_weights

# 添加通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        #Fex：：给每个特征通道生成一个权重值，通过两个全连接层构建通道间的相关性，输出的权重值数目和输入特征图的通道数相同。[1,1,c] ==> [1,1,c]
        # 修改FC层的输入输出维度
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征 [batch_size, seq_len, channels]
        Returns:
            out: 加权后的特征 [batch_size, seq_len, channels]
            weights: 通道注意力权重 [batch_size, 1, channels]
        """
        b, l, c = x.size()

        # 调整维度顺序以适应池化操作
        x_perm = x.permute(0, 2, 1)  # [batch_size, channels, seq_len]
        
        # Squeeze: 全局平均池化
        y = self.avg_pool(x_perm)  # [batch_size, channels, seq_len]==> [batch_size, channels, 1]  [256, 75, 1]

        y = y.squeeze(-1)       # [batch_size, channels]
        
        weights = self.fc(y) # [256, 75]
        # 调整权重维度用于广播
        weights = weights.unsqueeze(1)  # [batch_size, 1, channels]
        
        # 将权重应用到输入特征
        out = x * weights  # 广播乘法 [batch_size, seq_len, channels] [256, 100, 75]

        return out, weights


class MTAD_GAT(nn.Module):
    def __init__(self, device, input_size, n_features, window_size, args,
                 kernel_size=7, use_gatv2=True, feat_gat_embed_dim=None,
                 time_gat_embed_dim=None, dropout=0.2, alpha=0.2):
        super(MTAD_GAT, self).__init__()
        
        # 基础特征提取
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2)
        
        # 多尺度特征提取
        self.multi_scale_conv = nn.ModuleList([
            ConvLayer(n_features, k) for k in [3, 5, 7]
        ])
        

        # 为每种特征单独添加通道注意力
        # self.conv_attention = ChannelAttention(n_features)
        # self.feat_attention = ChannelAttention(n_features)
        # self.temp_attention = ChannelAttention(n_features)
        
        # 特征融合权重层
        # self.feature_weights = nn.Sequential(
        #     nn.Linear(n_features * 3, n_features),
        #     nn.ReLU(),
        #     nn.Linear(n_features, 3),
        #     nn.Softmax(dim=-1)
        # )
        self.fusion_attention = OptimizedFeatureFusion(n_features)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(n_features, n_features // 2),
            nn.ReLU(),
            nn.Linear(n_features // 2, n_features)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_features)

    def forward(self, x):
        # 输入形状调整
        shape = x.shape
        x = x.reshape((shape[0], shape[1], -1))
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, n_features]

        # 1. 多尺度特征提取
        multi_scale_features = [conv(x) for conv in self.multi_scale_conv]
        multi_scale_feature = torch.mean(torch.stack(multi_scale_features), dim=0)
        
        # 2. 基本特征提取
        x_conv = multi_scale_feature         # [batch_size, seq_len, n_features]
        h_feat = self.feature_gat(x_conv)    # 空间特征
        h_temp = self.temporal_gat(x_conv)   # 时间特征

        weighted_features, _ = self.fusion_attention(x_conv, h_feat, h_temp)

        features = self.layer_norm(weighted_features) 

        # Mamba序列建模
        # mamba_out = self.mamba(features)  # [batch_size, seq_len, n_features]
        
        output = self.output_layer(features)
        
        return output

    def get_attention_weights(self):
        """获取最近一次forward过程中的注意力权重"""
        return self.fusion_weights if hasattr(self, 'fusion_weights') else None