import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class DMoLE_Router(nn.Module):
    """Autoencoder Router for Continual Learning Task Selection (Text-Only Residual Version)"""
    def __init__(self, feature_dim=512, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.task_aes = nn.ModuleDict()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.register_buffer('base_feat', torch.zeros(1, feature_dim))
        self.has_base_feat = False

    def set_base_feature(self, base_feat_tensor):
        self.base_feat.copy_(base_feat_tensor.detach().view(1, -1))
        self.has_base_feat = True
        logger.info("D-MoLE Router: Base feature successfully registered for residual stripping.")

    def _process_features(self, features):
        features = features.to(torch.float32)
        if self.has_base_feat:
            features = features - self.base_feat.to(features.device)
        return F.normalize(features, p=2, dim=-1) * 10.0 
        
    def add_task(self, task_name):
        """[难点2: L_i 的架构] - autoencoder内部结构的设计"""
        self.task_aes[task_name] = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), 
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim), # 压缩到 64 维
            nn.GELU(),
            nn.Linear(self.latent_dim, self.hidden_dim), # 重构回高维
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feature_dim)
        ).to(torch.float32)
        
    def train_ae(self, task_name, features, epochs=50):
        """[难点5: Proxy Function Training] - 训练当前任务的 AE"""
        ae = self.task_aes[task_name].to(features.device)
        optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
        processed_features = self._process_features(features).detach()
        
        for _ in range(epochs):
            optimizer.zero_grad()
            recon = ae(processed_features)
            loss = F.mse_loss(recon, processed_features)
            loss.backward()
            optimizer.step()
            
    def get_top_k_experts(self, features, k=1, threshold=0.1):
        """
        [难点2: 指标选择] - 选择 MSE 作为 z_i
        [难点4: Routing] - 阈值 threshold 的选取与判定
        """
        if len(self.task_aes) == 0:
            return None
            
        errors = {}
        processed_features = self._process_features(features)
        
        with torch.no_grad():
            for name, ae in self.task_aes.items():
                ae = ae.to(processed_features.device)
                recon = ae(processed_features)
                errors[name] = F.mse_loss(recon, processed_features).item()
                
        logger.info(f"\n[Router Debug] MSE Errors against old tasks: {errors}")
                
        min_error_task = min(errors, key=errors.get)
        
        if errors[min_error_task] > threshold:
            logger.info(f"OOD Detected! Min error {errors[min_error_task]:.4f} > threshold {threshold}")
            return "fallback" # OOD detection
            
        return min_error_task