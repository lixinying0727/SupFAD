import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.eps = 1e-8
        
    def forward(self, features, anomaly_features, labels=None):
        # 特征归一化
        features = F.normalize(features, dim=-1)
        if anomaly_features.shape[-1] != features.shape[-1]:
            anomaly_features = anomaly_features.transpose(-1, -2)
        anomaly_features = F.normalize(anomaly_features, dim=-1)
        
        # 全局对比损失计算
        global_sim = torch.matmul(features, features.t())
        global_sim = torch.clamp(global_sim, min=-1.0, max=1.0)
        global_sim = global_sim / self.temperature
        
        # 计算全局损失
        pos_mask = torch.eye(features.shape[0], device=features.device)
        numerator = torch.exp(global_sim * pos_mask).sum(1)
        denominator = torch.exp(global_sim).sum(1) - torch.exp(global_sim).diag()
        global_loss = -torch.log((numerator + self.eps) / (denominator + self.eps))
        global_loss = global_loss.mean()

        # 修改局部对比损失计算
        try:
            # 计算全局-局部相似度
            global_local_sim = torch.matmul(
                features.unsqueeze(1),
                anomaly_features.transpose(-2, -1)
            )
            
            # 限制相似度范围并应用温度系数
            global_local_sim = torch.clamp(global_local_sim, min=-10.0, max=10.0)
            global_local_sim = global_local_sim / self.temperature
            
            # 平均池化并限制范围
            exp_sim = torch.exp(global_local_sim)
            mean_exp_sim = exp_sim.mean(dim=2)
            mean_exp_sim = torch.clamp(mean_exp_sim, min=self.eps, max=1.0-self.eps)
            
            # 计算局部损失
            local_loss = -torch.log(1 - mean_exp_sim + self.eps).mean()
            
            # 检查损失值
            if torch.isnan(local_loss) or torch.isinf(local_loss):
                print(f"Local loss computation details:")
                print(f"global_local_sim range: {global_local_sim.min():.4f} to {global_local_sim.max():.4f}")
                print(f"mean_exp_sim range: {mean_exp_sim.min():.4f} to {mean_exp_sim.max():.4f}")
                local_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
            
        except Exception as e:
            print(f"Error in local loss computation: {str(e)}")
            local_loss = torch.tensor(0.0, device=features.device, requires_grad=True)

        # 合并损失并添加权重
        total_loss = global_loss + 0.5 * local_loss  # 降低局部损失的权重
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # print(f"Warning: Invalid HCL loss detected! global_loss: {global_loss}, local_loss: {local_loss}")
            return torch.tensor(0.0, device=features.device, requires_grad=True)
            
        return total_loss

class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, smooth=1e-6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth
        
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] or [B, H, W]
            target: [B, H, W] or [B]
        """
        
        # 1. 处理预测掩码
        if len(pred.shape) == 4:  # [B, C, H, W]
            if pred.shape[1] > 1:
                # 如果有多个通道，取最后一个作为异常预测
                pred = pred[:, -1]  # [B, H, W]
            else:
                pred = pred.squeeze(1)  # [B, H, W]
                
        # 2. 处理目标掩码
        if len(target.shape) == 1:  # [B]
            B = target.shape[0]
            H, W = pred.shape[-2:]
            target = target.view(B, 1, 1).expand(-1, H, W).float()
        elif len(target.shape) == 2:  # [H, W]
            target = target.unsqueeze(0)  # [1, H, W]
        
        # 确保维度匹配
        assert pred.shape == target.shape, \
            f"Shape mismatch after processing: pred {pred.shape}, target {target.shape}"
            
        
        # 3. 应用 sigmoid 确保值域在 [0,1]
        pred = torch.sigmoid(pred)
        
        # 4. 展平所有空间维度
        B = pred.shape[0]
        pred_flat = pred.reshape(B, -1)     # [B, H*W]
        target_flat = target.reshape(B, -1)  # [B, H*W]
        
        
        # 5. 计算 Dice
        intersection = (pred_flat * target_flat).sum(1)  # [B]
        denominator = pred_flat.sum(1) + target_flat.sum(1)  # [B]
        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)  # [B]
        
        # 6. 应用 Focal 权重
        focal_weights = (1 - dice) ** self.gamma
        weighted_dice = self.alpha * focal_weights * dice
        
        # 7. 计算最终损失
        loss = -torch.log(weighted_dice + self.smooth).mean()
        
        return loss

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        
    def guassian_kernel(self, source, target):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(n_samples, n_samples, -1)
        total1 = total.unsqueeze(1).expand(n_samples, n_samples, -1)
        L2_distance = ((total0-total1)**2).sum(2)
        
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        """
        MMD Loss for domain adaptation
        Args:
            source: 源域特征 [B, D]
            target: 目标域特征 [B, D]
        """
        batch_size = min(source.size()[0], target.size()[0])
        kernels = self.guassian_kernel(source[:batch_size], target[:batch_size])
        
        # 计算 MMD 损失
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        loss = torch.mean(XX + YY - XY - YX)
        return loss

class TotalLoss(nn.Module):
    def __init__(self, lambda_mmd=0.1, l2_reg=0.01):
        super().__init__()
        self.hcl_loss = HierarchicalContrastiveLoss()
        self.dice_loss = FocalDiceLoss()
        self.mmd_loss = MMDLoss()
        self.lambda_mmd = lambda_mmd
        self.l2_reg = l2_reg
        
    def forward(self, features, anomaly_features, pred_mask, true_mask, source_domain, target_domain):
        """
        组合损失函数
        """
        # 分别计算各个损失并进行安全检查
        def safe_compute(loss_fn, *args):
            loss = loss_fn(*args)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid {loss_fn.__class__.__name__} detected!")
                return torch.tensor(0.0, device=features.device, requires_grad=True)
            return loss
            
        hcl_loss = safe_compute(self.hcl_loss, features, anomaly_features)
        dice_loss = safe_compute(self.dice_loss, pred_mask, true_mask)
        mmd_loss = safe_compute(self.mmd_loss, source_domain, target_domain)
        
        # 打印损失值以进行调试
        # print(f"Individual losses - HCL: {hcl_loss.item():.4f}, Dice: {dice_loss.item():.4f}, MMD: {mmd_loss.item():.4f}")
        
        total_loss = hcl_loss + dice_loss + self.lambda_mmd * mmd_loss
        return total_loss, {
            'contrast_loss': hcl_loss.item(),
            'dice_loss': dice_loss.item(),
            'mmd_loss': mmd_loss.item()
        }