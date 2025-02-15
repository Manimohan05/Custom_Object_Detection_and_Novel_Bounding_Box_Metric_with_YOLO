import numpy as np
import torch

def custom_bbox_similarity(box1, box2):
    """
    Compute a custom bounding box similarity score.
    Inputs:
        - box1, box2: Bounding boxes in [x1, y1, x2, y2] format
    Returns:
        - similarity score (0 to 1)
    """
    # Compute IoU
    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    iou = compute_iou(box1, box2)

    # Compute center distance penalty
    center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    center_dist = np.linalg.norm(np.array(center1) - np.array(center2))
    center_penalty = np.exp(-center_dist / 50)  # Normalize by 50 pixels

    # Compute aspect ratio similarity
    def aspect_ratio(box):
        return (box[2] - box[0]) / (box[3] - box[1]) if (box[3] - box[1]) > 0 else 1

    aspect_ratio_similarity = 1 - abs(aspect_ratio(box1) - aspect_ratio(box2)) / max(aspect_ratio(box1), aspect_ratio(box2))

    # Weighted combination
    similarity_score = (iou + center_penalty + aspect_ratio_similarity) / 3
    return similarity_score
