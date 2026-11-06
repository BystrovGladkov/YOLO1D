from typing import List, Tuple, Union

import numpy as np
import torch


def xw_to_x1x2(x: torch.Tensor) -> torch.Tensor:
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 1] / 2 
    y[:, 1] = x[:, 0] + x[:, 1] / 2 
    return y


def compute_metric(output: torch.Tensor, target: torch.Tensor,
                   iou_v: Union[torch.Tensor, List[float]]) -> torch.Tensor:
    """
    Вход
        output (n,4) - предсказания сети в формате [(x1, x2, conf, class)]
        target (m,3) - целевые значения в формате [(class, x1, x2)]
        iou_v (k) - пороги IoU для сопоставления. Как правило — [0.5, 0.95] с шагом 0.05
    Выход
        Матрица (n,k). Количество строк совпадает с output, а количество столбцов с размером iou_v. В матрице указано является ли предсказание модели TruePositive с каждым порогом IoU.
    """
    thresholds = torch.as_tensor(iou_v, dtype=output.dtype, device=output.device).flatten()
    correct = torch.zeros((output.shape[0], thresholds.numel()), dtype=torch.bool,
                          device=output.device)
    if output.shape[0] == 0 or target.shape[0] == 0 or thresholds.numel() == 0:
        return correct

    # Broadcasting [N, 1] against [1, M] builds every prediction-target pair.
    left = torch.maximum(output[:, None, 0], target[None, :, 1])
    right = torch.minimum(output[:, None, 1], target[None, :, 2])
    intersection = (right - left).clamp_min(0)
    output_length = (output[:, 1] - output[:, 0]).clamp_min(0)
    target_length = (target[:, 2] - target[:, 1]).clamp_min(0)
    union = output_length[:, None] + target_length[None, :] - intersection
    overlap = intersection / (union + 1e-7)
    same_class = output[:, 3:4] == target[None, :, 0]

    for column, threshold in enumerate(thresholds):
        # Transpose to [M, N], so each row is a target and each column a detection.
        pairs = torch.nonzero(
            (overlap.T >= threshold) & same_class.T, as_tuple=False
        )
        if pairs.shape[0] == 0:
            continue

        pair_iou = overlap[pairs[:, 1], pairs[:, 0]].detach().cpu().numpy()
        order = torch.as_tensor(np.argsort(pair_iou)[::-1].copy(), device=pairs.device)
        pairs = pairs[order]
        pair_array = pairs.cpu().numpy()
        # Resolve duplicate assignments by keeping the highest-IoU pair first.
        first_per_detection = np.unique(pair_array[:, 1], return_index=True)[1]
        pairs = pairs[torch.as_tensor(first_per_detection, device=pairs.device)]
        pair_array = pairs.cpu().numpy()
        first_per_target = np.unique(pair_array[:, 0], return_index=True)[1]
        target_indices = torch.as_tensor(first_per_target, device=pairs.device)
        correct[pairs[target_indices, 1], column] = True
    return correct


def suppress_contained(output, main_classes, secondary_classes, threshold=0.8):
    """Постобработка после NMS: удаление вторичных объектов, вложенных в главный.

    Объект вторичного класса удаляется, если доля его длины внутри объекта главного класса больше
    `threshold`, а уверенность главного объекта не ниже его собственной. Обход — по убыванию уверенности,
    как в NMS: удалённый объект сам ничего не удаляет; при равной уверенности старшим считается тот,
    что раньше в выдаче. Если класс и главный, и вторичный (например, стрелочный перевод), из двух
    вложенных объектов остаётся более уверенный — так убираются дубли, которые NMS по IoU пропускает
    (у вложенного короткого объекта IoU с длинным мал).

    output: (n, ≥4) [x1, x2, conf, cls, ...] — детекции одного сигнала; возвращается подмножество строк
    в исходном порядке.
    """
    n = output.shape[0]
    if n == 0 or not main_classes or not secondary_classes:
        return output
    order = torch.argsort(output[:, 2], descending=True, stable=True)
    det = output[order]
    cls = det[:, 3].long()
    is_main = torch.isin(cls, torch.as_tensor(list(main_classes), device=det.device))
    is_secondary = torch.isin(cls, torch.as_tensor(list(secondary_classes), device=det.device))
    if not is_main.any() or not is_secondary.any():
        return output
    x1, x2 = det[:, 0], det[:, 1]
    # Each matrix cell compares one interval (row) with another (column).
    inter = (torch.min(x2[:, None], x2[None, :]) - torch.max(x1[:, None], x1[None, :])).clamp(min=0)
    # Divide each row by its interval length to get the fraction inside each column.
    inside = inter / (x2 - x1).clamp(min=1e-7)[:, None]
    keep = torch.ones(n, dtype=torch.bool, device=det.device)
    for j in range(1, n):
        if is_secondary[j]:
            senior = keep[:j] & is_main[:j]
            if (inside[j, :j][senior] > threshold).any():
                keep[j] = False
    keep_original = torch.zeros(n, dtype=torch.bool, device=det.device)
    keep_original[order] = keep
    return output[keep_original]


def non_max_suppression(outputs: torch.Tensor, confidence_threshold: float = 0.001,
                        iou_threshold: float = 0.65) -> List[torch.Tensor]:
    max_det = 300
    max_nms = 30000
    batch_size = outputs.shape[0]
    num_classes = outputs.shape[1] - 2
    results = [torch.zeros((0, 4), device=outputs.device) for _ in range(batch_size)]

    for batch_index, prediction in enumerate(outputs):
        candidate = prediction.transpose(0, 1)
        candidate = candidate[candidate[:, 2:].amax(dim=1) > confidence_threshold]
        if candidate.shape[0] == 0:
            continue

        boxes = xw_to_x1x2(candidate[:, :2])
        class_scores = candidate[:, 2:]
        if num_classes > 1:
            box_indices, class_indices = torch.nonzero(
                class_scores > confidence_threshold, as_tuple=True
            )
            detections = torch.cat((
                boxes[box_indices],
                class_scores[box_indices, class_indices, None],
                class_indices[:, None].to(boxes.dtype),
            ), dim=1)
        else:
            scores, class_indices = class_scores.max(dim=1)
            detections = torch.cat((
                boxes, scores[:, None], class_indices[:, None].to(boxes.dtype)
            ), dim=1)
            detections = detections[scores > confidence_threshold]

        if detections.shape[0] == 0:
            continue

        order = torch.argsort(detections[:, 2], descending=True)[:max_nms]
        kept = []
        while order.numel() and len(kept) < max_det:
            selected = order[0]
            kept.append(selected)
            order = order[1:]
            if order.numel() == 0:
                break

            remaining = detections[order]
            same_class = remaining[:, 3] == detections[selected, 3]
            # Compare the selected interval with all remaining intervals at once.
            class_offset = detections[selected, 3] * 50000
            selected_left = detections[selected, 0] + class_offset
            selected_right = detections[selected, 1] + class_offset
            remaining_left = remaining[:, 0] + class_offset
            remaining_right = remaining[:, 1] + class_offset
            intersection = (
                torch.minimum(selected_right, remaining_right)
                - torch.maximum(selected_left, remaining_left)
            ).clamp_min(0)
            selected_length = (selected_right - selected_left).clamp_min(0)
            remaining_length = (remaining_right - remaining_left).clamp_min(0)
            union = selected_length + remaining_length - intersection
            iou = intersection / union.clamp_min(torch.finfo(union.dtype).tiny)
            order = order[~(same_class & (iou > iou_threshold))]

        results[batch_index] = detections[torch.stack(kept)]
    return results


def compute_ap(
    tp: np.ndarray,
    conf: np.ndarray,
    output: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tp = np.asarray(tp)
    conf = np.asarray(conf)
    output = np.asarray(output)
    target = np.asarray(target)
    if tp.ndim == 1:
        tp = tp[:, None]

    order = np.argsort(-conf)
    tp, conf, output = tp[order], conf[order], output[order]
    classes, target_counts = np.unique(target, return_counts=True)
    class_count = classes.size

    precision_curve = np.zeros((class_count, 1000))
    recall_curve = np.zeros((class_count, 1000))
    average_precision = np.zeros((class_count, tp.shape[1]))
    confidence_grid = np.linspace(0, 1, 1000)
    ap_grid = np.linspace(0, 1, 101)

    for class_index, class_id in enumerate(classes):
        detections = output == class_id
        if not detections.any():
            continue

        true_positive = tp[detections].astype(float).cumsum(axis=0)
        false_positive = (1 - tp[detections]).cumsum(axis=0)
        recall = true_positive / (target_counts[class_index] + eps)
        precision = true_positive / (true_positive + false_positive)

        # Interpolate class precision/recall at a shared confidence grid.
        recall_curve[class_index] = np.interp(
            -confidence_grid, -conf[detections], recall[:, 0], left=0
        )
        precision_curve[class_index] = np.interp(
            -confidence_grid, -conf[detections], precision[:, 0], left=1
        )

        for threshold_index in range(tp.shape[1]):
            recall_points = np.concatenate(([0.0], recall[:, threshold_index], [1.0]))
            precision_points = np.concatenate(([1.0], precision[:, threshold_index], [0.0]))
            # Enforce non-increasing precision before integrating the PR curve.
            precision_envelope = np.maximum.accumulate(precision_points[::-1])[::-1]
            interpolated = np.interp(ap_grid, recall_points, precision_envelope)
            average_precision[class_index, threshold_index] = np.sum(
                np.diff(ap_grid) * (interpolated[1:] + interpolated[:-1]) / 2
            )

    f1_curve = (
        2 * precision_curve * recall_curve
        / (precision_curve + recall_curve + eps)
    )
    mean_f1 = f1_curve.mean(axis=0)
    window_size = round(mean_f1.size * 0.1 * 2) // 2 + 1
    edge = np.full(window_size // 2, mean_f1[0])
    smoothed_f1 = np.convolve(
        np.concatenate((edge, mean_f1, edge)),
        np.ones(window_size) / window_size,
        mode="valid",
    )
    best_index = smoothed_f1.argmax()
    precision = precision_curve[:, best_index]
    recall = recall_curve[:, best_index]
    ap50 = average_precision[:, 0]
    ap = average_precision.mean(axis=1)
    return precision, recall, ap50, ap
