"""Router implementation for the original D-MoLE v1 pipeline."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DMoLE_Router(nn.Module):
    """Autoencoder router used by the original D-MoLE training flow."""

    def __init__(self, feature_dim=512, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.task_aes = nn.ModuleDict()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.register_buffer("base_feat", torch.zeros(1, feature_dim))
        self.has_base_feat = False

    def set_base_feature(self, base_feat_tensor):
        """Register the generic base prompt feature used for residual stripping."""
        self.base_feat.copy_(base_feat_tensor.detach().view(1, -1))
        self.has_base_feat = True
        logger.info(
            "D-MoLE Router: Base feature successfully registered for residual stripping."
        )

    def _process_features(self, features):
        """Convert raw features into normalized residual features."""
        features = features.to(torch.float32)
        if self.has_base_feat:
            features = features - self.base_feat.to(features.device)
        return F.normalize(features, p=2, dim=-1) * 10.0

    def add_task(self, task_name):
        """Register one task-specific autoencoder."""
        self.task_aes[task_name] = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        ).to(torch.float32)

    def train_ae(self, task_name, features, epochs=50):
        """Train the autoencoder assigned to one task."""
        autoencoder = self.task_aes[task_name].to(features.device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
        processed_features = self._process_features(features).detach()

        for _ in range(epochs):
            optimizer.zero_grad()
            reconstruction = autoencoder(processed_features)
            loss = F.mse_loss(reconstruction, processed_features)
            loss.backward()
            optimizer.step()

    def get_top_k_experts(self, features, k=1, threshold=0.1):
        """
        Route a prompt feature by reconstruction error.

        The legacy interface keeps the `k` argument for compatibility, but the
        current implementation performs top-1 routing only.
        """
        if len(self.task_aes) == 0:
            return None

        reconstruction_errors = {}
        processed_features = self._process_features(features)

        with torch.no_grad():
            for task_name, autoencoder in self.task_aes.items():
                autoencoder = autoencoder.to(processed_features.device)
                reconstruction = autoencoder(processed_features)
                reconstruction_errors[task_name] = F.mse_loss(
                    reconstruction, processed_features
                ).item()

        logger.info(
            "\n[Router Debug] MSE Errors against old tasks: %s",
            reconstruction_errors,
        )

        best_task = min(reconstruction_errors, key=reconstruction_errors.get)

        if reconstruction_errors[best_task] > threshold:
            logger.info(
                "OOD Detected! Min error %.4f > threshold %.4f",
                reconstruction_errors[best_task],
                threshold,
            )
            return "fallback"

        return best_task
