"""
Custom loss functions for online training in BBScore.

This module contains loss functions that can be used during online training
of linear probes, including correlation-based losses.
"""

import torch
import torch.nn as nn


def pearson_correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Negative Pearson correlation loss (to minimize).

    Computes the negative mean Pearson correlation across output dimensions.
    This loss encourages the model predictions to correlate with targets.

    Args:
        pred: Predictions tensor of shape (batch_size, n_outputs) or (batch_size,)
        target: Target tensor of shape (batch_size, n_outputs) or (batch_size,)

    Returns:
        Negative mean correlation (scalar tensor). Minimizing this maximizes correlation.
    """
    # Center predictions and targets (subtract mean along batch dimension)
    pred_c = pred - pred.mean(dim=0)
    target_c = target - target.mean(dim=0)

    # Compute numerator: sum of products along batch dimension
    num = (pred_c * target_c).sum(dim=0)

    # Compute denominator: product of norms, clamped to avoid division by zero
    denom = (pred_c.norm(dim=0) * target_c.norm(dim=0)).clamp(min=1e-6)

    # Pearson correlation per output dimension
    corr = num / denom

    # Return negative mean correlation (we minimize, so negative = maximize correlation)
    return -corr.mean()


def compute_scale_ratio(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the scale ratio between predictions and targets.

    This is useful for monitoring whether the model's predictions have the
    correct scale relative to the targets. A ratio close to 1.0 indicates
    good scale calibration.

    Args:
        preds: Predictions tensor of shape (batch_size, n_outputs) or (batch_size,)
        targets: Target tensor of shape (batch_size, n_outputs) or (batch_size,)

    Returns:
        Scale ratio: mean(preds.std) / mean(targets.std)
    """
    return preds.std(dim=0).mean() / targets.std(dim=0).mean().clamp(min=1e-6)


def ccc_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Concordance Correlation Coefficient (CCC) loss.

    CCC measures agreement between predictions and targets, combining both
    correlation AND scale matching into a single metric. This naturally prevents
    the scale collapse problem without needing separate loss terms.

    CCC = (2 * cov(pred, target)) / (var(pred) + var(target) + (mean(pred) - mean(target))^2)

    Properties:
    - CCC = 1: Perfect agreement (correlation=1, same scale, same mean)
    - CCC = 0: No agreement
    - CCC < 0: Negative agreement
    - High Pearson but collapsed scale -> Low CCC (this is the key benefit!)

    Args:
        pred: Predictions tensor (batch, features)
        target: Target tensor (batch, features)
        eps: Small value for numerical stability

    Returns:
        Scalar loss tensor (1 - mean CCC across features)
    """
    # Compute means per feature
    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)

    # Compute variances per feature (unbiased)
    pred_var = pred.var(dim=0, unbiased=True)
    target_var = target.var(dim=0, unbiased=True)

    # Compute covariance per feature
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    covar = (pred_centered * target_centered).mean(dim=0)

    # CCC formula per feature
    numerator = 2 * covar
    denominator = pred_var + target_var + (pred_mean - target_mean) ** 2 + eps
    ccc = numerator / denominator

    # Return 1 - mean CCC (minimizing this maximizes CCC)
    return 1 - ccc.mean()


class CCCLoss(nn.Module):
    """
    Concordance Correlation Coefficient loss module.

    Recommended for neural encoding tasks as it naturally handles scale preservation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ccc_loss(pred, target)


class CombinedMSECCCLoss(nn.Module):
    """
    Combined loss using MSE and CCC.

    Combines absolute error (MSE) with concordance (CCC) for cases where
    both precise values and scale preservation matter.

    Args:
        ccc_weight: Weight for the CCC loss component (default: 0.5)
    """

    def __init__(self, ccc_weight: float = 0.5):
        super().__init__()
        self.ccc_weight = ccc_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = self.mse_loss(pred, target)
        ccc = ccc_loss(pred, target)
        return (1 - self.ccc_weight) * mse + self.ccc_weight * ccc


class CombinedMSECorrelationLoss(nn.Module):
    """
    Combined loss that uses both MSE and negative Pearson correlation.

    This loss function combines the traditional MSE loss with a correlation
    loss component. The correlation loss helps ensure that predictions
    maintain proper correlation with targets, while MSE ensures scale accuracy.

    Args:
        correlation_weight: Weight for the correlation loss component (default: 0.5)
    """

    def __init__(self, correlation_weight: float = 0.5):
        super().__init__()
        self.correlation_weight = correlation_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined MSE + correlation loss.

        Args:
            pred: Predictions tensor
            target: Target tensor

        Returns:
            Combined loss value
        """
        mse = self.mse_loss(pred, target)
        corr_loss = pearson_correlation_loss(pred, target)
        return mse + self.correlation_weight * corr_loss
