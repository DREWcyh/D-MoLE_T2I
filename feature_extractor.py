"""Feature extraction utilities for the legacy v1 D-MoLE router."""

import torch
import torch.nn.functional as F


def extract_and_fuse_features(t_input, latents=None):
    """
    Extract the v1 routing feature.

    Notes
    -----
    The historical function name is kept for API compatibility with the
    training and inference scripts. In the current v1 implementation it returns
    the normalized text feature only, even though it still computes the
    auxiliary vision branch when latents are available.
    """
    device = t_input.device
    dtype = t_input.dtype
    batch_size = t_input.size(0)

    # Pool token features into a single prompt feature and project it to 512D.
    pooled_text = t_input.max(dim=1)[0]

    rng_devices = []
    if device.type == "cuda" and device.index is not None:
        rng_devices = [device.index]

    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(42)
        projection = torch.randn((4096, 512), device=device, dtype=torch.float32)
        projection, _ = torch.linalg.qr(projection)
        projection = projection.to(dtype)

    text_features = torch.mm(pooled_text, projection)

    if latents is not None:
        pooled_latents = F.adaptive_max_pool2d(latents, (11, 11))
        vision_features = pooled_latents.view(batch_size, -1)
    else:
        vision_features = torch.randn(
            (text_features.size(0), 484),
            device=text_features.device,
            dtype=torch.float32,
        )

    normalized_text = F.normalize(text_features, p=2, dim=-1, eps=1e-8)
    normalized_vision = F.normalize(vision_features, p=2, dim=-1, eps=1e-8)

    # Keep the legacy fusion step visible for documentation purposes.
    fused_features = torch.cat([normalized_text, normalized_vision], dim=-1)
    _ = fused_features

    return normalized_text
