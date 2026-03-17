"""Router implementation for the v3 residual prototype pipeline."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ResidualPrototypeRouter(nn.Module):
    """
    Tuning-Free Residual Prototype Router for task-agnostic continual T2I routing.

    Core idea
    ---------
    1. Use a global base feature to strip away generic prompt semantics.
    2. Convert each text feature into a normalized residual fingerprint.
    3. Store one prototype per task directly, without any optimizer or training loop.
    4. Route inference prompts by cosine distance to saved prototypes.

    Parameters
    ----------
    feature_dim : int, default=512
        Dimensionality of the input text feature space.
    """

    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

        self.register_buffer("base_feat", torch.zeros(1, feature_dim, dtype=torch.float32))
        self.register_buffer(
            "_has_base_feat_flag",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

        # Kept as a Python bool to match the requested API.
        self.has_base_feat = False
        self.task_prototypes = nn.ParameterDict()

    def _sync_base_feat_flag(self):
        """Keep the Python-side `has_base_feat` flag aligned with the persisted buffer flag."""
        self.has_base_feat = bool(self._has_base_feat_flag.item())

    def _ensure_task_placeholder(self, task_name):
        """Register an empty prototype slot so state_dict loading can populate it."""
        if task_name not in self.task_prototypes:
            self.task_prototypes[task_name] = nn.Parameter(
                torch.zeros(1, self.feature_dim, device=self.base_feat.device, dtype=torch.float32),
                requires_grad=False,
            )

    def _coerce_feature_tensor(self, tensor, tensor_name):
        """
        Normalize feature tensor shape and dtype.

        Accepts tensors with shape `[D]` or `[B, D]` and always returns `[B, D]`
        in `torch.float32`.
        """
        if not torch.is_tensor(tensor):
            raise TypeError(f"{tensor_name} must be a torch.Tensor, got {type(tensor)!r}")

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise ValueError(
                f"{tensor_name} must have shape [D] or [B, D], got {tuple(tensor.shape)}"
            )

        if tensor.shape[-1] != self.feature_dim:
            raise ValueError(
                f"{tensor_name} last dimension must be {self.feature_dim}, "
                f"got {tensor.shape[-1]}"
            )

        if tensor.shape[0] == 0:
            raise ValueError(f"{tensor_name} must contain at least one sample")

        return tensor.detach().to(dtype=torch.float32)

    def set_base_feature(self, base_feat_tensor):
        """
        Register the global base feature used for residual stripping.

        Parameters
        ----------
        base_feat_tensor : torch.Tensor
            Tensor with shape `[1, feature_dim]` or `[feature_dim]`.
        """
        base_feat_tensor = self._coerce_feature_tensor(base_feat_tensor, "base_feat_tensor")

        if base_feat_tensor.shape[0] != 1:
            raise ValueError(
                "base_feat_tensor must describe exactly one base feature, "
                f"got batch size {base_feat_tensor.shape[0]}"
            )

        with torch.no_grad():
            self.base_feat.copy_(
                base_feat_tensor.to(device=self.base_feat.device, dtype=torch.float32)
            )
            self._has_base_feat_flag.fill_(True)

        self.has_base_feat = True
        logger.info(
            "ResidualPrototypeRouter: base feature registered with shape %s",
            tuple(self.base_feat.shape),
        )

    def _get_residual(self, feat):
        """
        Convert a text feature into a normalized residual feature.

        Steps
        -----
        1. Cast to `torch.float32` for stable distance computation.
        2. Subtract the global base feature if available.
        3. L2-normalize along the last dimension.
        """
        self._sync_base_feat_flag()

        feat = self._coerce_feature_tensor(feat, "feat")

        if self.has_base_feat:
            feat = feat - self.base_feat.to(device=feat.device, dtype=torch.float32)

        return F.normalize(feat, p=2, dim=-1)

    def add_task(self, task_name, task_text_feat):
        """
        Register a new task prototype without any training or optimization.

        Parameters
        ----------
        task_name : str
            Name of the task, e.g. `"stage1"`.
        task_text_feat : torch.Tensor
            Text feature for the task prompt with shape `[1, feature_dim]` or
            `[feature_dim]`. If a batched tensor is provided, residuals are averaged
            into a single prototype for robustness.
        """
        if not task_name:
            raise ValueError("task_name must be a non-empty string")

        with torch.no_grad():
            prototype = self._get_residual(task_text_feat)

            if prototype.shape[0] != 1:
                logger.info(
                    "ResidualPrototypeRouter: task '%s' received %d features; averaging them into one prototype.",
                    task_name,
                    prototype.shape[0],
                )
                prototype = F.normalize(prototype.mean(dim=0, keepdim=True), p=2, dim=-1)

            prototype = prototype.to(device=self.base_feat.device, dtype=torch.float32)
            self._ensure_task_placeholder(task_name)
            self.task_prototypes[task_name].data.copy_(prototype.detach().clone())

        logger.info(
            "ResidualPrototypeRouter: saved prototype for task '%s' with shape %s",
            task_name,
            tuple(self.task_prototypes[task_name].shape),
        )

    @torch.no_grad()
    def get_top_k_experts(self, t_feat, threshold=0.15, k=1):
        """
        Route a test text feature to the nearest residual prototype.

        Parameters
        ----------
        t_feat : torch.Tensor
            Test text feature with shape `[1, feature_dim]` or `[feature_dim]`.
            Batched input with shape `[B, feature_dim]` is also accepted; distances
            are averaged across the batch.
        threshold : float, default=0.15
            OOD threshold on the minimum cosine distance.
        k : int, default=1
            Reserved for interface compatibility. This router performs top-1 routing.

        Returns
        -------
        str or None
            Best matching task name, `"fallback"` if the best distance exceeds the
            threshold, or `None` if no tasks are registered.
        """
        if len(self.task_prototypes) == 0:
            logger.info("ResidualPrototypeRouter: no task prototypes registered, returning None.")
            return None

        if k != 1:
            logger.info(
                "ResidualPrototypeRouter performs top-1 routing only; received k=%s and will ignore it.",
                k,
            )

        infer_res = self._get_residual(t_feat)

        distances = {}
        best_task = None
        min_dist = None

        for task_name, prototype in self.task_prototypes.items():
            task_prototype = prototype.to(device=infer_res.device, dtype=torch.float32)
            expanded_prototype = task_prototype.expand_as(infer_res)
            cosine_distance = 1.0 - F.cosine_similarity(
                infer_res,
                expanded_prototype,
                dim=-1,
            )
            distance_value = cosine_distance.mean().item()
            distances[task_name] = distance_value

            if min_dist is None or distance_value < min_dist:
                min_dist = distance_value
                best_task = task_name

        formatted_distances = ", ".join(
            f"'{name}': {dist:.4f}" for name, dist in distances.items()
        )
        logger.info("🔍 [Router Debug] Cosine Distances: {%s}", formatted_distances)

        if min_dist is None:
            return None

        if min_dist > threshold:
            logger.warning(
                "ResidualPrototypeRouter: OOD detected, min cosine distance %.4f > threshold %.4f. Returning fallback.",
                min_dist,
                threshold,
            )
            return "fallback"

        return best_task

    def load_state_dict(self, state_dict, strict=True):
        """
        Load router state and lazily register missing prototype keys before loading.
        """
        prototype_prefix = "task_prototypes."
        for key in state_dict.keys():
            if key.startswith(prototype_prefix):
                task_name = key[len(prototype_prefix):].split(".", 1)[0]
                self._ensure_task_placeholder(task_name)

        result = super().load_state_dict(state_dict, strict=strict)
        self._sync_base_feat_flag()
        return result
