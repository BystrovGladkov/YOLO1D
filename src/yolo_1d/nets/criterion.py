import torch

from yolo_1d.nets.ops import compute_iou, make_anchors
from torch.nn.functional import cross_entropy
from yolo_1d.nets.nn import YOLO1D

class TaskAlignedAssigner(torch.nn.Module):
    def __init__(
        self,
        nc: int = 80,
        top_k: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
    ):
        """Configure the number of classes and alignment-score selection."""
        super().__init__()
        self.top_k = top_k
        self.nc = nc
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        batch_size, anchor_count = pd_scores.shape[0], pd_bboxes.shape[1]
        gt_count = gt_bboxes.shape[1]

        if gt_count == 0:
            return (
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
            )

        inside_mask = self._anchors_inside_targets(anc_points, gt_bboxes)
        candidate_mask = (inside_mask * mask_gt).bool()
        overlap_matrix = torch.zeros(
            (batch_size, gt_count, anchor_count),
            dtype=pd_bboxes.dtype,
            device=pd_bboxes.device,
        )
        class_score_matrix = torch.zeros(
            (batch_size, gt_count, anchor_count),
            dtype=pd_scores.dtype,
            device=pd_scores.device,
        )

        batch_indices = torch.arange(batch_size, device=pd_scores.device)[:, None]
        class_indices = gt_labels.squeeze(-1).long()
        all_class_scores = pd_scores[batch_indices, :, class_indices]
        class_score_matrix[candidate_mask] = all_class_scores[candidate_mask]

        predicted_candidates = pd_bboxes[:, None].expand(-1, gt_count, -1, -1)
        target_candidates = gt_bboxes[:, :, None].expand(-1, -1, anchor_count, -1)
        overlap_matrix[candidate_mask] = compute_iou(
            target_candidates[candidate_mask],
            predicted_candidates[candidate_mask],
        ).squeeze(-1).clamp_(0)

        alignment = class_score_matrix.pow(self.alpha) * overlap_matrix.pow(self.beta)
        positive_mask = self._select_top_anchors(alignment, inside_mask, mask_gt)
        positive_mask, foreground = self._resolve_multiple_targets(
            positive_mask, overlap_matrix
        )
        target_indices = positive_mask.argmax(dim=1)

        selected_boxes, selected_labels = self._gather_assigned_targets(
            target_indices, gt_bboxes, gt_labels
        )
        class_targets = self._make_class_targets(selected_labels, foreground)
        class_targets = self._normalize_class_targets(
            class_targets, alignment, overlap_matrix, positive_mask
        )
        return selected_boxes, class_targets, foreground.bool()
 
    def _anchors_inside_targets(self, anchors, boxes):
        left = boxes[..., 0:1]
        right = boxes[..., 1:2]
        anchor_x = anchors[:, 0].view(1, 1, -1)
        distance_to_edges = torch.stack(
            (anchor_x - left, right - anchor_x), dim=-1
        )
        return distance_to_edges.amin(dim=-1).gt_(self.eps)

    def _select_top_anchors(self, alignment, inside_mask, mask_gt):
        _, best_indices = torch.topk(
            alignment, self.top_k, dim=-1, largest=True
        )
        valid_targets = mask_gt.expand(-1, -1, self.top_k).bool()
        best_indices.masked_fill_(~valid_targets, 0)

        selected = torch.zeros(
            alignment.shape, dtype=torch.int8, device=alignment.device
        )
        selected.scatter_add_(
            -1,
            best_indices,
            torch.ones_like(best_indices, dtype=torch.int8),
        )
        selected.masked_fill_(selected > 1, 0)
        return selected.to(alignment.dtype) * inside_mask * mask_gt

    @staticmethod
    def _resolve_multiple_targets(positive_mask, overlaps):
        foreground = positive_mask.sum(dim=1)
        if foreground.max() > 1:
            multiple_assignments = (foreground[:, None] > 1).expand_as(positive_mask)
            best_target = overlaps.argmax(dim=1)
            preferred = torch.zeros_like(positive_mask)
            preferred.scatter_(1, best_target[:, None], 1)
            positive_mask = torch.where(
                multiple_assignments, preferred, positive_mask
            ).float()
            foreground = positive_mask.sum(dim=1)
        return positive_mask, foreground

    @staticmethod
    def _gather_assigned_targets(target_indices, boxes, labels):
        batch_size, gt_count = boxes.shape[:2]
        flat_indices = target_indices + (
            torch.arange(batch_size, device=labels.device)[:, None] * gt_count
        )
        selected_boxes = boxes.reshape(-1, boxes.shape[-1])[flat_indices]
        selected_labels = labels.long().reshape(-1)[flat_indices].clamp_min_(0)
        return selected_boxes, selected_labels

    def _make_class_targets(self, labels, foreground):
        class_targets = torch.zeros(
            (*labels.shape, self.nc), dtype=torch.int64, device=labels.device
        )
        class_targets.scatter_(2, labels.unsqueeze(-1), 1)
        active = foreground.unsqueeze(-1).expand_as(class_targets) > 0
        return torch.where(active, class_targets, 0)

    def _normalize_class_targets(
        self, class_targets, alignment, overlaps, positive_mask
    ):
        positive_alignment = alignment * positive_mask
        max_alignment = positive_alignment.amax(dim=-1, keepdim=True)
        max_overlap = (overlaps * positive_mask).amax(dim=-1, keepdim=True)
        scaled_alignment = (
            positive_alignment * max_overlap / (max_alignment + self.eps)
        ).amax(dim=1, keepdim=True).transpose(1, 2)
        return class_targets * scaled_alignment


class BoxDFLLosses(torch.nn.Module):
    def __init__(self, dfl_ch):
        super().__init__()
        self.dfl_ch = dfl_ch

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        positive_weights = target_scores.sum(dim=-1)[fg_mask].unsqueeze(-1)

        matched_iou = compute_iou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask]
        )
        localization_loss = (
            (1.0 - matched_iou) * positive_weights
        ).sum() / target_scores_sum

        left_edges, right_edges = target_bboxes.chunk(2, dim=-1)
        edge_distances = torch.cat(
            (anchor_points - left_edges, right_edges - anchor_points), dim=-1
        ).clamp(0, self.dfl_ch - 0.01)
        selected_distribution = pred_dist[fg_mask].view(
            -1, self.dfl_ch + 1
        )
        distribution_loss = self.df_loss(
            selected_distribution, edge_distances[fg_mask]
        )
        distribution_loss = (
            distribution_loss * positive_weights
        ).sum() / target_scores_sum

        return localization_loss, distribution_loss

    @staticmethod
    def df_loss(pred_dist, target):
        lower_bin = target.long()
        upper_bin = lower_bin + 1
        lower_weight = upper_bin - target
        upper_weight = 1.0 - lower_weight

        lower_ce = cross_entropy(
            pred_dist, lower_bin.reshape(-1), reduction="none"
        ).view_as(target)
        upper_ce = cross_entropy(
            pred_dist, upper_bin.reshape(-1), reduction="none"
        ).view_as(target)
        interpolated_ce = lower_ce * lower_weight + upper_ce * upper_weight
        return interpolated_ce.mean(dim=-1, keepdim=True)


class CalculateLoss:
    def __init__(self, model: YOLO1D, weight_box: float, weight_cls: float, weight_dfl: float):
        if hasattr(model, "module"):
            model = model.module

        device = next(model.parameters()).device
        head = model.head

        self.weight_box = weight_box
        self.weight_cls = weight_cls
        self.weight_dfl = weight_dfl
        self.stride = head.stride
        self.num_classes = head.num_classes
        self.no = head.num_outputs
        self.reg_max = head.reg_max
        self.device = device

        self.box_loss = BoxDFLLosses(self.reg_max - 1).to(device)
        self.cls_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(nc=self.num_classes, top_k=10, alpha=0.5, beta=4.0)
        self.project = torch.arange(self.reg_max, dtype=torch.float, device=device)

    def box_decode(self, anchor_points, pred_dist):
        batch, anchors, channels = pred_dist.shape
        distributions = pred_dist.view(batch, anchors, 2, channels // 2)
        probabilities = distributions.softmax(dim=-1)
        projected = probabilities.matmul(self.project.type(probabilities.dtype))
        left_distances, right_distances = projected.chunk(2, dim=-1)
        starts = anchor_points - left_distances
        ends = anchor_points + right_distances
        return torch.cat((starts, ends), dim=-1)

    def _flatten_head_outputs(self, outputs):
        batch_size = outputs[0].shape[0]
        flattened_levels = [
            level.view(batch_size, self.no, -1) for level in outputs
        ]
        combined = torch.cat(flattened_levels, dim=2)
        distributions, class_logits = combined.split(
            (self.reg_max * 2, self.num_classes), dim=1
        )
        class_logits = class_logits.permute(0, 2, 1).contiguous()
        distributions = distributions.permute(0, 2, 1).contiguous()
        return distributions, class_logits

    def _pad_targets(self, targets, batch_size, input_size):
        packed = torch.cat(
            (
                targets["idx"].view(-1, 1),
                targets["cls"].view(-1, 1),
                targets["box"],
            ),
            dim=1,
        ).to(self.device)

        if packed.shape[0] == 0:
            return torch.zeros(batch_size, 0, 3, device=self.device)

        batch_ids = packed[:, 0]
        _, counts = batch_ids.unique(return_counts=True)
        max_targets = counts.max()
        padded = torch.zeros(batch_size, max_targets, 3, device=self.device)
        for batch_index in range(batch_size):
            selection = batch_ids == batch_index
            if selection.any():
                padded[batch_index, :selection.sum()] = packed[selection, 1:]

        center_width = padded[..., 1:3].mul_(input_size[[0, 0]])
        centers = center_width[..., 0].clone()
        half_width = center_width[..., 1] / 2
        padded[..., 1] = centers - half_width
        padded[..., 2] = centers + half_width
        return padded

    def __call__(self, outputs, targets):
        pred_distri, pred_scores = self._flatten_head_outputs(outputs)
        batch_size = pred_scores.shape[0]
        input_size = (
            torch.tensor(
                outputs[0].shape[2:],
                device=self.device,
                dtype=pred_scores.dtype,
            )
            * self.stride[0]
        )
        anchor_points, stride_tensor = make_anchors(
            outputs, self.stride, offset=0.5
        )

        ground_truth = self._pad_targets(targets, batch_size, input_size)
        gt_labels, gt_bboxes = ground_truth.split((1, 2), dim=2)
        valid_gt = gt_bboxes.sum(dim=2, keepdim=True).gt_(0)
        pred_bboxes = self.box_decode(anchor_points, pred_distri)

        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            valid_gt,
        )

        normalizer = max(target_scores.sum(), 1)
        cls_loss = self.cls_loss(
            pred_scores, target_scores.to(pred_scores.dtype)
        ).sum() / normalizer
        box_loss = torch.zeros(1, device=self.device)
        dfl_loss = torch.zeros(1, device=self.device)

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            box_loss, dfl_loss = self.box_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                normalizer,
                fg_mask,
            )

        return (
            box_loss * self.weight_box,
            cls_loss * self.weight_cls,
            dfl_loss * self.weight_dfl,
        )


# ---------------------------------------------------------------------------------------------- режим MC (M7)
def flatten_lead_targets(targets: dict, n_leads: int):
    """Цели батча записей → цели «батча отведений» (строка b·n_leads + k — отведение k примера b).

    targets: cls (n, 1), box (n, 2), idx (n,) — пример, lead (n,) — отведение, valid (B, n_leads) — отведения
    в лоссе/метриках. Возвращает (targets с idx по отведениям, row_valid (B·n_leads,) bool).
    """
    idx = targets["idx"].view(-1).long() * n_leads + targets["lead"].view(-1).long()
    valid = targets["valid"].reshape(-1).bool()
    return {"cls": targets["cls"], "box": targets["box"], "idx": idx.to(targets["idx"].dtype)}, valid


class MultiLeadLoss:
    """Лосс режима MC (M7, R5 §2.3): исходный `ComputeLoss` (TAL, DFL, BCE, веса), посчитанный для каждого
    отведения отдельно и просуммированный по отведениям.

    Выход головы с n_leads наборами — «батч отведений» (B·n_leads, no, A) (`Head.forward`), поэтому каждое
    отведение — отдельная строка: ассайнер и все слагаемые для разных отведений независимы. Отведения
    с valid = False (непригодная разметка, R5 §2.2) из лосса исключаются.

    norm
        "joint" — сумма по отведениям, делённая на суммарный target_scores батча (масштаб как в режиме S
        с батчем из B·n_leads отведений); "per_lead" — лосс каждого отведения нормируется отдельно, затем сумма.
    """

    def __init__(self, criterion: CalculateLoss, n_leads: int, norm: str = "joint"):
        if norm not in ("joint", "per_lead"):
            raise ValueError(f"norm должен быть 'joint' или 'per_lead', получено {norm!r}")
        self.criterion, self.n_leads, self.norm = criterion, n_leads, norm

    def _select(self, outputs, targets, rows: torch.Tensor):
        """Строки `rows` (bool по батчу отведений) и цели этих строк с перенумерованными idx."""
        new_index = torch.cumsum(rows.long(), 0) - 1
        idx = targets["idx"].long()
        keep = rows[idx.to(rows.device)].cpu()
        sel = {"cls": targets["cls"][keep], "box": targets["box"][keep],
               "idx": new_index.cpu()[idx[keep]].to(targets["idx"].dtype)}
        return [o[rows.to(o.device)] for o in outputs], sel

    def __call__(self, outputs, targets):
        flat, valid = flatten_lead_targets(targets, self.n_leads)
        if self.norm == "joint":
            return self.criterion(*self._select(outputs, flat, valid))
        total = None
        lead_of_row = torch.arange(valid.numel()) % self.n_leads
        for k in range(self.n_leads):
            rows = valid & (lead_of_row == k)
            if not rows.any():
                continue
            losses = self.criterion(*self._select(outputs, flat, rows))
            total = losses if total is None else tuple(a + b for a, b in zip(total, losses))
        return total
