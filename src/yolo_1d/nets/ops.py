from typing import List, Tuple

import torch


def make_anchors(
    x: List[torch.Tensor],
    head_strides: torch.Tensor,
    offset: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create 1D anchor centers and matching strides for each feature-map level."""
    if not x:
        raise ValueError("at least one feature map is required")
    if any(feature.ndim != 3 for feature in x):
        raise ValueError("feature maps must have shape (batch, channels, width)")

    reference = x[0]
    widths = torch.tensor(
        [feature.shape[-1] for feature in x],
        device=reference.device,
        dtype=torch.long,
    )
    strides = torch.as_tensor(
        head_strides, device=reference.device, dtype=reference.dtype
    ).reshape(-1)
    if strides.numel() != len(x):
        raise ValueError(
            f"expected one stride per feature map ({len(x)}), got {strides.numel()}"
        )

    anchors_count = int(widths.sum().item())
    level_ids = torch.repeat_interleave(
        torch.arange(len(x), device=reference.device), widths
    )
    level_starts = torch.cumsum(widths, dim=0) - widths
    local_indices = torch.arange(anchors_count, device=reference.device)
    local_indices = local_indices - level_starts[level_ids]

    # Each level restarts its grid at zero; level_ids also selects its stride.
    anchors = (local_indices.to(dtype=reference.dtype) + offset).unsqueeze(-1)
    stride_tensor = strides[level_ids].unsqueeze(-1)
    return anchors, stride_tensor


def compute_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Return 1D DIoU similarity for broadcastable intervals ``(..., 2)``.

    Since an interval has no height, the CIoU aspect-ratio term is undefined
    in 1D. This uses IoU with the normalized center-distance penalty instead.
    """
    start1, end1 = box1.unbind(dim=-1)
    start2, end2 = box2.unbind(dim=-1)

    # Broadcast interval endpoints to compare each requested pair.
    intersection = (
        torch.minimum(end1, end2) - torch.maximum(start1, start2)
    ).clamp_min(0)
    length1 = (end1 - start1).clamp_min(0)
    length2 = (end2 - start2).clamp_min(0)
    union = (length1 + length2 - intersection).clamp_min(eps)
    iou = intersection / union

    center1 = (start1 + end1) / 2
    center2 = (start2 + end2) / 2
    center_distance_sq = (center1 - center2).square()
    enclosing_span = (
        torch.maximum(end1, end2) - torch.minimum(start1, start2)
    ).clamp_min(0)
    distance_penalty = center_distance_sq / (enclosing_span.square() + eps)
    return (iou - distance_penalty).unsqueeze(-1)
