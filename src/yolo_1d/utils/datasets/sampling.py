import numpy as np

from yolo_1d.utils.datasets.dataset import Dataset


def get_data_weights(dataset: Dataset, eps=1e-5):
    """ Calculates a weights of data samples to fix unbalanced classes. """
    data_to_class = []
    class_counts = np.zeros(dataset.num_classes, dtype=np.float32)
    for labels in dataset.labels.values():
        data_to_class.append(labels[:, 0].astype(int))
        for i in labels[:, 0]:
            class_counts[int(i)] += 1
            
    inv_freq = 1.0 / (class_counts + eps)  
    weights = []
    for classes in data_to_class:
        if len(classes) == 0:
            weights.append(1.0)
        else:
            w = float(inv_freq[classes].max())  # или .mean()
            weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    # нормируем, чтобы средний вес ~1
    weights = weights / (weights.mean() + eps)
    return weights
