import torch
import torch.nn.functional as F

def extract_and_fuse_features(t_input, latents=None):
    """
    [难点1: 特征构建 - 常规池化对等版]
    目标：使用确定性的最大池化，使图像特征维度显著提升，达到与文本对等。
    """
    device = t_input.device
    dtype = t_input.dtype
    batch_size = t_input.size(0)

    # --- 1. 文本特征降维 (4096 -> 512) ---
    # 使用固定的随机正交投影 (不可训练，保证距离保持)
    t_raw = t_input.max(dim=1)[0] # [B, 4096]
    
    rng_devices = []
    if device.type == "cuda" and device.index is not None:
        rng_devices = [device.index]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(42) 
        # 核心修复：强制使用 float32 生成随机矩阵
        proj_matrix = torch.randn((4096, 512), device=device, dtype=torch.float32)
        # QR 分解也必须在 float32 下进行
        proj_matrix, _ = torch.linalg.qr(proj_matrix)
        # 计算完成后，将投影矩阵转回原始 dtype (例如 fp16)
        proj_matrix = proj_matrix.to(dtype)

    t_feat = torch.mm(t_raw, proj_matrix) # [B, 512]

    # --- 2. 图像特征提取 (VAE 4 -> 484) ---
    if latents is not None:
        # [难点1 突破点]：使用常规的 AdaptiveMaxPool2d
        # 将 [B, 4, H, W] 池化为 [B, 4, 11, 11]
        # 11x11 网格能捕捉图像中 121 个局部区域的最强信号
        v_pool = F.adaptive_max_pool2d(latents, (11, 11)) 
        v_feat = v_pool.view(batch_size, -1) # 展平得到 4 * 121 = 484 维
    else:
        # [难点5: 模态缺失]
        v_feat = torch.randn((t_feat.size(0), 484), device=t_feat.device, dtype=torch.float32)

    # --- 3. 归一化与平衡拼接 ---
    # 分别归一化，确保在计算余弦相似度时，两者的“拉力”是一样的
    t_feat_norm = F.normalize(t_feat, p=2, dim=-1, eps=1e-8) # [B, 512]
    v_feat_norm = F.normalize(v_feat, p=2, dim=-1, eps=1e-8) # [B, 484]

    # 最终输出 [B, 996] 的 1D 向量
    z_feat_batch = torch.cat([t_feat_norm, v_feat_norm], dim=-1)
    
    # return z_feat_batch
    return t_feat_norm # 仅返回文本特征
