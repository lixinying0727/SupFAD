import torch
import torch.nn as nn

class CrossLayerFeatureAggregator(nn.Module):
    def __init__(self, feature_dim, num_layers=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # 为每层特征学习权重
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # 特征转换层
        self.transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features_list):
        # features_list: 包含4个层的特征 [f6, f12, f18, f24]
        weights = torch.softmax(self.layer_weights, dim=0)
        
        transformed_features = []
        for i, feat in enumerate(features_list):
            # 转换每层特征
            trans_feat = self.transform(feat)
            # 应用权重
            weighted_feat = trans_feat * weights[i]
            transformed_features.append(weighted_feat)
        
        # 加权融合
        aggregated_feature = sum(transformed_features)
        return aggregated_feature, weights
