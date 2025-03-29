from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
import numpy as np
from skimage import measure

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    # 创建一个与 amaps 形状相同的布尔型数组，初始化为 False
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    # 获取 amaps 的最小值和最大值
    min_th, max_th = amaps.min(), amaps.max()
    # 计算阈值的步长
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    # 从最小阈值到最大阈值，以 delta 为步长遍历
    for th in np.arange(min_th, max_th, delta):
        # 将小于等于阈值的元素设为 0，大于阈值的元素设为 1
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        # 遍历 binary_amaps 和 masks 中的元素
        for binary_amap, mask in zip(binary_amaps, masks):
            # 计算每个区域的属性
            for region in measure.regionprops(measure.label(mask)):
                # 计算真阳性像素的数量
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                # 计算真阳性像素占区域面积的比例
                pro.append(tp_pixels / region.area)
        # 计算反向掩码（1 - 原始掩码）
        inverse_masks = 1 - masks
        # 计算假阳性像素的数量
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        # 计算假阳性率
        fpr = fp_pixels / inverse_masks.sum()
        # 将真阳性像素比例的平均值添加到 pros 列表
        pros.append(np.array(pro).mean())
        # 将假阳性率添加到 fprs 列表
        fprs.append(fpr)
        # 将当前阈值添加到 ths 列表
        ths.append(th)
    # 将列表转换为 numpy 数组
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    # 找到满足期望假阳性率的索引
    idxes = fprs < expect_fpr
    # 筛选出满足期望假阳性率的假阳性率
    fprs = fprs[idxes]
    # 对筛选后的假阳性率进行归一化
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    # 计算曲线下面积（AUC）
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def image_level_metrics(results, obj, metric):
    # 从 results 字典中获取 obj 对应的 'gt_sp' 键的值作为真实标签
    gt = results[obj]['gt_sp']
    # 从 results 字典中获取 obj 对应的 'pr_sp' 键的值作为预测结果
    pr = results[obj]['pr_sp']
    # 将真实标签转换为 numpy 数组
    gt = np.array(gt)
    # 将预测结果转换为 numpy 数组
    pr = np.array(pr)
    # 如果度量指标是 'image-auroc'
    if metric == 'image-auroc':
        # 计算 ROC 曲线下面积（Area Under the Receiver Operating Characteristic Curve）
        performance = roc_auc_score(gt, pr)
    # 如果度量指标是 'image-ap'
    elif metric == 'image-ap':
        # 计算平均精度（Average Precision）
        performance = average_precision_score(gt, pr)
    # 返回性能指标结果
    return performance
    # table.append(str(np.round(performance * 100, decimals=1)))


def pixel_level_metrics(results, obj, metric):
    # 从 results 字典中获取 obj 对应的 'imgs_masks' 键的值作为真实标签
    gt = results[obj]['imgs_masks']
    # 从 results 字典中获取 obj 对应的 'anomaly_maps' 键的值作为预测结果
    pr = results[obj]['anomaly_maps']
    # 将真实标签转换为 numpy 数组
    gt = np.array(gt)
    # 将预测结果转换为 numpy 数组
    pr = np.array(pr)
    # 如果度量指标是 'pixel-auroc'
    if metric == 'pixel-auroc':
        # 计算像素级别的 ROC 曲线下面积（Area Under the Receiver Operating Characteristic Curve）
        # 使用 ravel 函数将多维数组展平为一维数组
        performance = roc_auc_score(gt.ravel(), pr.ravel())
    # 如果度量指标是 'pixel-aupro'
    elif metric == 'pixel-aupro':
        # 如果 gt 是四维数组，移除第一个维度
        if len(gt.shape) == 4:
            gt = gt.squeeze(1)
        # 如果 pr 是四维数组，移除第一个维度
        if len(pr.shape) == 4:
            pr = pr.squeeze(1)
        # 调用自定义的 cal_pro_score 函数计算性能指标
        performance = cal_pro_score(gt, pr)
    # 返回性能指标结果
    return performance
    