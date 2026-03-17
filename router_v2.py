"""Router implementation for the v2 cross-modal projection pipeline."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossModalRouter(nn.Module):
    """
    Cross-Modal Projection Router for task-agnostic continual T2I routing.

    核心思想:
    1. 每个任务维护一个轻量级文本->视觉映射器 mapper。
    2. 训练阶段:
       - 计算并保存该任务真实视觉特征的质心。
       - 用 MSE 训练该任务 mapper, 使文本特征能够投影到视觉特征空间。
    3. 推理阶段:
       - 对给定文本特征, 使用所有历史任务 mapper 预测视觉特征。
       - 将预测视觉特征与各任务保存的真实视觉质心做余弦距离比较。
       - 选择距离最小的任务; 若最小距离超过阈值, 返回 "fallback"。

    Parameters
    ----------
    text_dim : int, default=512
        Dimension of the text feature space.
    vision_dim : int, default=484
        Dimension of the vision feature space.
    hidden_dim : int, default=256
        Hidden dimension used by each task-specific MLP mapper.
    """

    def __init__(self, text_dim=512, vision_dim=484, hidden_dim=256):
        super().__init__()
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim

        self.task_mappers = nn.ModuleDict()
        self.task_centroids = nn.ParameterDict()

        # Non-persistent buffer used only to track the router device cleanly.
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    def _router_device(self):
        """Return the current device where newly created modules/tensors should live."""
        return self._device_anchor.device

    def _coerce_feature_tensor(self, tensor, expected_last_dim, tensor_name):
        """
        Normalize feature tensor shape and dtype.

        Accepts either `[D]` or `[B, D]` and always returns `[B, D]` in float32.
        """
        if not torch.is_tensor(tensor):
            raise TypeError(f"{tensor_name} must be a torch.Tensor, got {type(tensor)!r}")

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise ValueError(
                f"{tensor_name} must have shape [D] or [B, D], got {tuple(tensor.shape)}"
            )

        if tensor.shape[-1] != expected_last_dim:
            raise ValueError(
                f"{tensor_name} last dimension must be {expected_last_dim}, "
                f"got {tensor.shape[-1]}"
            )

        return tensor.detach().to(dtype=torch.float32)

    def add_task(self, task_name):
        """
        Register a new task-specific cross-modal mapper.

        Parameters
        ----------
        task_name : str
            Name of the task, e.g. "stage1" or "task_dog".
        """
        if not task_name:
            raise ValueError("task_name must be a non-empty string")

        if task_name in self.task_mappers:
            logger.info("CrossModalRouter: task '%s' already exists, mapper will be replaced.", task_name)

        mapper = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.vision_dim),
        ).to(device=self._router_device(), dtype=torch.float32)

        self.task_mappers[task_name] = mapper

        # Store a placeholder centroid so the router state_dict is self-contained.
        centroid = nn.Parameter(
            torch.zeros(1, self.vision_dim, device=self._router_device(), dtype=torch.float32),
            requires_grad=False,
        )
        self.task_centroids[task_name] = centroid

    def train_mapper(self, task_name, t_feats, v_feats, epochs=50):
        """
        Train one task-specific mapper and save the task visual centroid.

        Parameters
        ----------
        task_name : str
            Registered task name.
        t_feats : torch.Tensor
            Text features with shape `[B, text_dim]` or `[text_dim]`.
        v_feats : torch.Tensor
            Vision features with shape `[B, vision_dim]` or `[vision_dim]`.
        epochs : int, default=50
            Number of optimization epochs over the provided feature tensors.

        Returns
        -------
        torch.Tensor
            The saved visual centroid with shape `[1, vision_dim]`.
        """
        if task_name not in self.task_mappers:
            raise KeyError(
                f"Task '{task_name}' is not registered. Call add_task('{task_name}') first."
            )

        if epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {epochs}")

        mapper = self.task_mappers[task_name]
        mapper_device = next(mapper.parameters()).device

        t_feats = self._coerce_feature_tensor(t_feats, self.text_dim, "t_feats").to(mapper_device)
        v_feats = self._coerce_feature_tensor(v_feats, self.vision_dim, "v_feats").to(mapper_device)

        if t_feats.shape[0] != v_feats.shape[0]:
            raise ValueError(
                "t_feats and v_feats must have the same batch size, "
                f"got {t_feats.shape[0]} and {v_feats.shape[0]}"
            )

        if t_feats.shape[0] == 0:
            raise ValueError("t_feats/v_feats must contain at least one sample")

        # Step 1: compute and save the real visual centroid for this task.
        v_centroid = v_feats.mean(dim=0, keepdim=True)
        with torch.no_grad():
            if task_name not in self.task_centroids:
                self.task_centroids[task_name] = nn.Parameter(
                    v_centroid.clone(), requires_grad=False
                )
            else:
                self.task_centroids[task_name].data.copy_(v_centroid)

        # Step 2: fit text -> vision projection for the current task.
        mapper.train()
        optimizer = torch.optim.Adam(mapper.parameters(), lr=1e-3)

        last_loss = None
        for _ in range(epochs):
            optimizer.zero_grad()
            predicted_v = mapper(t_feats)
            loss = F.mse_loss(predicted_v, v_feats)
            loss.backward()
            optimizer.step()
            last_loss = loss.detach().item()

        mapper.eval()
        logger.info(
            "CrossModalRouter: trained mapper for task '%s' with centroid shape %s, final loss=%.6f",
            task_name,
            tuple(v_centroid.shape),
            0.0 if last_loss is None else last_loss,
        )
        return self.task_centroids[task_name].detach().clone()

    @torch.no_grad()
    def get_top_k_experts(self, t_feat, threshold=0.1, k=1):
        """
        Route a text feature to the closest task expert using cosine distance.

        Parameters
        ----------
        t_feat : torch.Tensor
            Single text feature with shape `[1, text_dim]` or `[text_dim]`.
            Batched input with shape `[B, text_dim]` is also accepted; in that case
            cosine distances are averaged across the batch.
        threshold : float
            OOD threshold on the minimum cosine distance.
        k : int, default=1
            Reserved for interface compatibility with the old router. The current
            implementation always performs top-1 routing.

        Returns
        -------
        str or None
            Best matching task name, `"fallback"` if the best distance is larger
            than `threshold`, or `None` if no task has been registered.
        """
        if len(self.task_mappers) == 0:
            logger.info("CrossModalRouter: no tasks registered, returning None.")
            return None

        if k != 1:
            logger.info(
                "CrossModalRouter currently performs top-1 routing only; received k=%s and will ignore it.",
                k,
            )

        distances = {}
        best_task = None
        best_distance = None

        processed_t_feat = self._coerce_feature_tensor(t_feat, self.text_dim, "t_feat")

        for task_name, mapper in self.task_mappers.items():
            mapper_device = next(mapper.parameters()).device
            task_t_feat = processed_t_feat.to(mapper_device)

            predicted_v_feat = mapper(task_t_feat).to(dtype=torch.float32)
            saved_centroid = self.task_centroids[task_name].to(mapper_device, dtype=torch.float32)

            cosine_distance = 1.0 - F.cosine_similarity(
                predicted_v_feat,
                saved_centroid.expand_as(predicted_v_feat),
                dim=-1,
            )
            distance_value = cosine_distance.mean().item()
            distances[task_name] = distance_value

            if best_distance is None or distance_value < best_distance:
                best_distance = distance_value
                best_task = task_name

        logger.info("Cosine Distances against old tasks: %s", distances)

        if best_distance is None:
            return None

        if best_distance > threshold:
            logger.info(
                "OOD detected in CrossModalRouter: min cosine distance %.6f > threshold %.6f",
                best_distance,
                threshold,
            )
            return "fallback"

        return best_task
