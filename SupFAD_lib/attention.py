import torch
import torch.nn as nn
import torch.nn.functional as F

class TextureAwareWindowAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # 1. 改进纹理分析器 - 使用更多尺度的卷积
        self.texture_analyzer = nn.ModuleDict({
            'conv3x3': nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            'conv5x5': nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            'conv7x7': nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        })
        
        # 2. 改进窗口注意力 - 使用可变形窗口
        self.window_sizes = [2, 4, 8]  # 多尺度窗口
        self.window_attns = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads) 
            for _ in range(len(self.window_sizes))
        ])
        
        # 3. 添加通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        
        # 4. 自适应融合模块
        self.fusion = nn.Sequential(
            nn.Linear(dim * len(self.window_sizes), dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, len(self.window_sizes))
        )
        
        # 5. 添加残差连接的权重
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # 6. 多尺度特征聚合
        self.feature_fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.ReLU()
        )

    def forward(self, x):
        import math
        
        # 保存输入的副本用于残差连接
        identity = x
        
        # 重塑输入便于处理
        L, B, D = x.shape
        x = x.permute(1, 0, 2)  # [B, L, D]
        cls_token, patches = x[:, :1, :], x[:, 1:, :]
        H = W = int(math.sqrt(patches.shape[1]))
        
        # 1. 多尺度纹理特征提取
        patches_4d = patches.reshape(B, H, W, D).permute(0, 3, 1, 2)
        texture_feats = []
        for conv in self.texture_analyzer.values():
            feat = conv(patches_4d)
            texture_feats.append(feat)
        
        # 融合纹理特征
        texture_feat = self.feature_fusion(
            torch.cat([f.permute(0,2,3,1).reshape(B, H*W, D) for f in texture_feats], dim=-1)
        )
        
        # 2. 多尺度窗口注意力
        window_outputs = []
        for i, window_size in enumerate(self.window_sizes):
            # 计算padding
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            
            if pad_h > 0 or pad_w > 0:
                padded = F.pad(patches_4d, (0, pad_w, 0, pad_h))
            else:
                padded = patches_4d
            
            padded_h, padded_w = padded.shape[2], padded.shape[3]
            
            # 应用窗口注意力
            window_tokens = self._apply_sliding_window(padded, window_size)
            
            # 确保维度正确 [seq_len, batch, dim]
            window_tokens = window_tokens.permute(1, 0, 2)
            
            # 应用多头注意力
            attn_out = self.window_attns[i](
                query=window_tokens,
                key=window_tokens,
                value=window_tokens
            )[0]
            
            # 恢复维度顺序 [batch, seq_len, dim]
            attn_out = attn_out.permute(1, 0, 2)
            
            # 重建特征图
            restored = self._restore_window_shape(attn_out, padded_h, padded_w, D, window_size)
            if pad_h > 0 or pad_w > 0:
                restored = restored[:, :H, :W, :]
            window_outputs.append(restored.reshape(B, H*W, D))
        
        # 3. 自适应融合
        fusion_weights = F.softmax(self.fusion(torch.cat(window_outputs, dim=-1)), dim=-1)
        output = sum(out * w.unsqueeze(-1) for out, w in zip(window_outputs, fusion_weights.unbind(-1)))
        
        # 4. 残差连接和归一化
        output = output + texture_feat  # 添加纹理特征
        output = self.gamma * output + (1 - self.gamma) * patches  # 可学习的残差连接
        
        # 重新添加CLS token
        output = torch.cat([cls_token, output], dim=1)  # [B, L, D]
        output = output.permute(1, 0, 2)  # [L, B, D]
        
        return output

    def _apply_sliding_window(self, x, window_size):
        """优化的滑动窗口处理"""
        B, C, H, W = x.shape
        
        # 确保尺寸能被窗口大小整除
        assert H % window_size == 0 and W % window_size == 0, \
            f"Feature map size ({H},{W}) must be divisible by window size {window_size}"
        
        # 窗口切分
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B, H//ws, W//ws, C, ws, ws]
        x = x.view(-1, window_size * window_size, C)  # [B*num_windows, ws*ws, C]
        
        return x

    def _restore_window_shape(self, x, H, W, C, window_size):
        """优化的形状恢复"""
        B = x.shape[0] // ((H * W) // (window_size * window_size))
        
        # 重塑为原始窗口布局
        x = x.view(B, H // window_size, W // window_size, window_size, window_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, H//ws, ws, W//ws, ws]
        x = x.view(B, C, H, W)  # [B, C, H, W]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        
        return x

class LocalGlobalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=7):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        
        # 局部注意力
        self.local_attn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=window_size, padding=window_size//2, groups=dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        # 全局注意力
        self.global_attn = nn.MultiheadAttention(dim, num_heads)
        
        # 自适应权重
        self.weight_learner = nn.Sequential(
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        B, N, C = x.shape
        
        # 局部注意力
        x_local = x.reshape(B, int(N**0.5), int(N**0.5), C)
        x_local = self.local_attn(x_local.permute(0,3,1,2))
        x_local = x_local.permute(0,2,3,1).reshape(B, N, C)
        
        # 全局注意力
        x_global = self.global_attn(x, x, x)[0]
        
        # 自适应权重
        weights = self.weight_learner(x.mean(1))
        
        # 加权融合
        x = weights[:,0:1].unsqueeze(1) * x_local + weights[:,1:2].unsqueeze(1) * x_global
        return x