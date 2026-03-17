"""Feature extraction utilities for the v3 residual prototype router."""

import torch


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


def _pool_text_features(t_input):
    """
    Pool token-level text encoder features into one prompt-level feature.

    Parameters
    ----------
    t_input : torch.Tensor
        Text encoder hidden states with shape `[B, L, D]`.
    """
    if t_input.ndim != 3:
        raise ValueError(f"t_input must have shape [B, L, D], got {tuple(t_input.shape)}")

    return t_input.max(dim=1)[0]


def extract_text_features(t_input, output_dim=512):
    """
    Extract projected text features for the tuning-free residual prototype router.

    This function intentionally returns the raw projected prompt feature without
    residual subtraction or normalization. Those geometric operations belong to
    `ResidualPrototypeRouter`, which applies the same base-feature stripping logic
    consistently during task registration and inference.

    Parameters
    ----------
    t_input : torch.Tensor
        Text encoder hidden states with shape `[B, L, D]`.
    output_dim : int, default=512
        Output text embedding dimension after deterministic projection.

    Returns
    -------
    torch.Tensor
        Projected text features with shape `[B, output_dim]` in `torch.float32`.
    """
    t_raw = _pool_text_features(t_input)
    proj = _build_orthogonal_projection(
        input_dim=t_raw.shape[-1],
        output_dim=output_dim,
        device=t_raw.device,
        dtype=t_raw.dtype,
    )
    return torch.mm(t_raw, proj).to(dtype=torch.float32)
