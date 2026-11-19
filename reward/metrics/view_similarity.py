"""Similarity of two objects' view embeddings (3DHarnessBench image_similarity.py)."""
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


def view_similarity(prediction: NDArray[np.float32], target: NDArray[np.float32]) -> dict[str, float]:
    """Compare unit view embeddings of two objects, shape (V, D) each.

    view_paired: mean cosine similarity of views with the same index.
    best_assignment: mean cosine similarity under the best one-to-one view matching,
    which compensates yaw offsets between the two objects.
    """
    cosine = prediction @ target.T
    rows, columns = linear_sum_assignment(-cosine)
    return {
        'view_paired': float(np.mean(np.diag(cosine))),
        'best_assignment': float(cosine[rows, columns].mean()),
    }
