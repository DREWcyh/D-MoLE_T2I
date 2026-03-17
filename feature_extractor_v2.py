import torch
import torch.nn.functional as F


def _build_orthogonal_projection(input_dim, output_dim, device, dtype, seed=42):
    """
    Build a deterministic random orthogonal projection matrix.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    output_dim : int
        Output projected dimension.
    device : torch.device
        Device where the matrix should be created.
    dtype : torch.dtype
        Final dtype used by downstream projection.
    seed : int, default=42
        Random seed for reproducibility.
    """
    rng_devices = []
    if device.type == "cuda" and device.index is not None:
        rng_devices = [device.index]

    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        proj = torch.randn((input_dim, output_dim), device=device, dtype=torch.float32)
        proj, _ = torch.linalg.qr(proj)
        proj = proj.to(dtype=dtype)
    return proj


def _project_text_features(t_input, output_dim=512):
    """
    Project pooled text features from T5 hidden states into a lower-dimensional space.

    Parameters
    ----------
    t_input : torch.Tensor
        Text encoder hidden states with shape `[B, L, D]`.
    output_dim : int, default=512
        Output text embedding dimension after deterministic projection.

    Returns
    -------
    torch.Tensor
        Projected text features with shape `[B, output_dim]`.
    """
    if t_input.ndim != 3:
        raise ValueError(f"t_input must have shape [B, L, D], got {tuple(t_input.shape)}")

    device = t_input.device
    dtype = t_input.dtype

    # Pool sequence features into a single prompt representation.
    t_raw = t_input.max(dim=1)[0]
    proj = _build_orthogonal_projection(
        input_dim=t_raw.shape[-1],
        output_dim=output_dim,
        device=device,
        dtype=dtype,
    )
    return torch.mm(t_raw, proj)


def _align_base_features(base_feat, target_batch_size):
    """
    Align base prompt features to the same batch size as the current prompt features.
    """
    if base_feat.shape[0] == target_batch_size:
        return base_feat
    if base_feat.shape[0] == 1:
        return base_feat.expand(target_batch_size, -1)
    raise ValueError(
        "base_t_input must have batch size 1 or match the current prompt batch size, "
        f"got base batch {base_feat.shape[0]} and target batch {target_batch_size}"
    )


def extract_text_features(t_input, output_dim=512, base_t_input=None):
    """
    Extract normalized text features from T5 hidden states.

    If `base_t_input` is provided, the returned text feature is the normalized
    residual feature:

        projected(prompt) - projected(base_prompt)

    This is used to encode the information difference between the original prompt
    and a generic base prompt.

    Parameters
    ----------
    t_input : torch.Tensor
        Text encoder hidden states with shape `[B, L, D]`.
    output_dim : int, default=512
        Output text embedding dimension after deterministic projection.
    base_t_input : torch.Tensor or None, default=None
        Base-prompt hidden states with shape `[1, L, D]` or `[B, L, D]`.

    Returns
    -------
    torch.Tensor
        Normalized text features with shape `[B, output_dim]`.
    """
    t_feat = _project_text_features(t_input, output_dim=output_dim)
    if base_t_input is not None:
        base_feat = _project_text_features(base_t_input, output_dim=output_dim)
        base_feat = _align_base_features(base_feat, t_feat.shape[0]).to(
            device=t_feat.device, dtype=t_feat.dtype
        )
        t_feat = t_feat - base_feat

    return F.normalize(t_feat.to(dtype=torch.float32), p=2, dim=-1, eps=1e-8)


def extract_vision_features(latents, pooled_hw=(11, 11)):
    """
    Extract normalized vision features from VAE latents.

    Parameters
    ----------
    latents : torch.Tensor
        Latent tensor with shape `[B, C, H, W]`.
    pooled_hw : tuple[int, int], default=(11, 11)
        Spatial size after adaptive max pooling.

    Returns
    -------
    torch.Tensor
        Normalized vision features with shape `[B, C * pooled_hw[0] * pooled_hw[1]]`.
    """
    if latents is None:
        raise ValueError("latents must not be None when extracting vision features")
    if latents.ndim != 4:
        raise ValueError(f"latents must have shape [B, C, H, W], got {tuple(latents.shape)}")

    batch_size = latents.shape[0]
    v_pool = F.adaptive_max_pool2d(latents, pooled_hw)
    v_feat = v_pool.reshape(batch_size, -1)
    return F.normalize(v_feat.to(dtype=torch.float32), p=2, dim=-1, eps=1e-8)


def extract_cross_modal_features(t_input, latents, text_dim=512, pooled_hw=(11, 11), base_t_input=None):
    """
    Extract text and vision features for the CrossModalRouter pipeline.

    Parameters
    ----------
    t_input : torch.Tensor
        Text encoder hidden states with shape `[B, L, D]`.
    latents : torch.Tensor
        Latent tensor with shape `[B, C, H, W]`.
    text_dim : int, default=512
        Output text embedding dimension.
    pooled_hw : tuple[int, int], default=(11, 11)
        Adaptive pooling size for vision features.
    base_t_input : torch.Tensor or None, default=None
        Base-prompt hidden states. If provided, text features are computed as
        the normalized residual between the current prompt and the base prompt.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        `(text_features, vision_features)`.
    """
    text_features = extract_text_features(
        t_input,
        output_dim=text_dim,
        base_t_input=base_t_input,
    )
    vision_features = extract_vision_features(latents, pooled_hw=pooled_hw)
    return text_features, vision_features
