import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import os

def compute_metrics(target, predicted_probs, threshold=0.5):
    """
    Compute accuracy and F1 score for binary segmentation.

    Parameters:
        target (torch.Tensor): Ground truth binary mask tensor of size [1, 1, H, W].
        predicted_probs (torch.Tensor): Predicted probability map tensor of size [1, 1, H, W].
        threshold (float): Threshold for binarizing the predicted probability map.

    Returns:
        accuracy (float): Accuracy score.
        f1 (float): F1 score.
    """
    # Binarize the predicted probabilities
    predicted_binary = (predicted_probs > threshold).float()

    target_flat = target.view(-1)
    predicted_flat = predicted_binary.view(-1)

    correct_predictions = (target_flat == predicted_flat).float()
    accuracy = correct_predictions.mean().item()
    
    return np.round(accuracy,3)

def compute_f1_score(target_masks, predicted_masks, threshold=0.5):
    """
    Compute F1 score for a batch of target and predicted masks.

    Args:
    - target_masks (torch.Tensor): Batch of target masks (ground truth).
    - predicted_masks (torch.Tensor): Batch of predicted masks.
    - threshold (float): Threshold for binarizing the predicted masks.

    Returns:
    - f1_score (float): Computed F1 score.
    """
    predicted_masks = (predicted_masks > threshold).float()

    target_flat = target_masks.view(-1)
    predicted_flat = predicted_masks.view(-1)

    tp = torch.sum(target_flat * predicted_flat)
    fp = torch.sum((1 - target_flat) * predicted_flat)
    fn = torch.sum(target_flat * (1 - predicted_flat))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return f1_score.item()