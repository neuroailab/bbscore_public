from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR, ReduceLROnPlateau
from tqdm import tqdm
import itertools

from metrics.base_online import OnlineMetric
from extractor_wrapper_online import OnlineFeatureExtractor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

import wandb


class OnlineLinearClassifier(OnlineMetric):
    def __init__(
        self,
        num_classes: int,
        input_feature_dim: int,
        lr_options: Optional[List[float]] = None,
        wd_options: Optional[List[float]] = None,
        n_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        ceiling: Optional[np.ndarray] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            input_feature_dim=input_feature_dim,
            internal_model_type="linear",
            lr_options=lr_options if lr_options else [1e-5, 1e-4, 1e-3],
            wd_options=wd_options if wd_options else [1e-5, 1e-4, 1e-3],
            n_epochs=n_epochs,
            patience=patience,
            batch_size=batch_size,
            device=device,
            ceiling=ceiling,
            task_type="classification",
        )

    def compute_raw(
        self,
        extractor: OnlineFeatureExtractor,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        return self.train_and_evaluate(extractor, train_dataloader, val_dataloader, test_dataloader)


class OnlineTransformerClassifier(OnlineMetric):
    def __init__(
        self,
        num_classes: int,
        input_feature_dim: int,  # This is D_fe from the feature extractor
        embed_dim: int = 256,   # Internal embedding dim for Transformer
        num_heads: int = 4,
        num_encoder_layers: int = 2,  # For AttentiveClassifier depth
        lr_options: Optional[List[float]] = None,
        wd_options: Optional[List[float]] = None,
        n_epochs: int = 50,
        patience: int = 10,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        ceiling: Optional[np.ndarray] = None,
        scheduler_type: str = "wsd",
    ):
        internal_model_params = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_encoder_layers": num_encoder_layers
        }
        super().__init__(
            num_classes=num_classes,
            # Passed as input_dim to TransformerInternalModel
            input_feature_dim=input_feature_dim,
            internal_model_type="transformer",
            internal_model_params=internal_model_params,
            lr_options=lr_options if lr_options else [
                1e-3],  # [1e-4, 5e-5],
            wd_options=wd_options if wd_options else [0],
            n_epochs=n_epochs,
            patience=patience,
            batch_size=batch_size,
            device=device,
            ceiling=ceiling,
            task_type="classification",
            scheduler_type=scheduler_type,
        )

    def compute_raw(
        self,
        extractor: OnlineFeatureExtractor,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        # The transformer model expects sequential input, so the extractor's sequence_mode matters.
        # Ensure extractor.sequence_mode is "all" or that features are appropriately shaped.
        if extractor.sequence_mode == "last" or extractor.sequence_mode == "concatenate":
            print(f"Warning: OnlineTransformerClassifier is being used with extractor sequence_mode='{extractor.sequence_mode}'. "
                  "This might not be optimal as TransformerInternalModel expects sequential input (B, T, D). "
                  "The TransformerInternalModel will unsqueeze the time dimension if features are (B, D).")
        return self.train_and_evaluate(extractor, train_dataloader, val_dataloader, test_dataloader)


class OnlineTransformerRegressor(OnlineMetric):
    def __init__(
        self,
        input_feature_dim: int,
        output_dim: int = 1,  # Number of target variables for regression
        embed_dim: int = 256,
        num_heads: int = 12,
        num_encoder_layers: int = 2,
        lr_options: Optional[List[float]] = None,
        wd_options: Optional[List[float]] = None,
        n_epochs: int = 10,
        patience: int = 10,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        ceiling: Optional[np.ndarray] = None,
        # 'deterministic_scheduler', 'warmup_stable_decay', # Option for scheduler
        scheduler_type: str = 'wsd',
    ):
        internal_model_params = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_encoder_layers": num_encoder_layers
        }
        super().__init__(
            num_classes=output_dim,  # For OnlineMetric's internal_output_dim logic
            input_feature_dim=input_feature_dim,
            internal_model_type="transformer",
            internal_model_params=internal_model_params,
            lr_options=lr_options if lr_options else [3e-4, 1e-3],
            wd_options=wd_options if wd_options else [1e-3, 1e-2],
            n_epochs=n_epochs,
            patience=patience,
            batch_size=batch_size,
            device=device,
            ceiling=ceiling,
            task_type="regression",
            scheduler_type=scheduler_type,
        )
        # Override internal_output_dim for regression specifically
        self.internal_output_dim = output_dim

    def compute_raw(
        self,
        extractor: OnlineFeatureExtractor,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        if extractor.sequence_mode == "last" or extractor.sequence_mode == "concatenate":
            print(f"Warning: OnlineTransformerRegressor is being used with extractor sequence_mode='{extractor.sequence_mode}'. "
                  "TransformerInternalModel will unsqueeze the time dimension if features are (B, D).")
        return self.train_and_evaluate(extractor, train_dataloader, val_dataloader, test_dataloader)


class OnlineNeuralTransformerRegressor(OnlineMetric):
    def __init__(
        self,
        input_feature_dim: int,
        output_dim: int = 1,  # Number of target variables for regression
        embed_dim: int = 252,
        num_heads: int = 12,
        num_encoder_layers: int = 2,
        lr_options: Optional[List[float]] = None,
        wd_options: Optional[List[float]] = None,
        n_epochs: int = 10,
        patience: int = 10,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        ceiling: Optional[np.ndarray] = None,
        # 'deterministic_scheduler', 'warmup_stable_decay', # Option for scheduler
        scheduler_type: str = 'wsd',
    ):
        internal_model_params = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_encoder_layers": num_encoder_layers
        }
        super().__init__(
            num_classes=output_dim,  # For OnlineMetric's internal_output_dim logic
            input_feature_dim=input_feature_dim,
            internal_model_type="transformer",
            internal_model_params=internal_model_params,
            lr_options=lr_options if lr_options else [3e-4, 1e-3],
            wd_options=wd_options if wd_options else [1e-3, 1e-2],
            n_epochs=n_epochs,
            patience=patience,
            batch_size=batch_size,
            device=device,
            ceiling=ceiling,
            task_type="regression",
            scheduler_type=scheduler_type,
            feature_normalization="none",
        )
        # Override internal_output_dim for regression specifically
        self.internal_output_dim = output_dim
        self._mean = None
        self._std = None

    @torch.no_grad()
    def _precompute_normalization_stats(self, extractor, dataloader):
        """
        Computes mean and std for feature normalization in a single pass over the training set.
        """
        n = 0
        mean = None
        M2 = None

        print("Pre-computing global normalization stats...")
        # Ensure model is in eval mode for extraction
        extractor.model.eval()

        for batch_data, _ in tqdm(dataloader, desc="Calculating Norm Stats"):
            # Extract features
            if isinstance(batch_data, dict):
                feats = extractor.extract_features_for_batch(batch_data)
            else:
                feats = extractor.extract_features_for_batch(
                    batch_data.to(self.device))

            # Flatten time dimension if present: (B, T, D) -> (B*T, D) for statistics
            X = feats.reshape(-1, feats.shape[-1])

            # Per-batch stats
            m = X.shape[0]
            batch_mean = X.mean(dim=0)
            # Sum of squared deviations (M2) inside the batch
            batch_M2 = ((X - batch_mean) ** 2).sum(dim=0)

            if mean is None:
                mean = batch_mean.clone()
                M2 = batch_M2.clone()
                n = m
                continue

            # Merge running stats with batch stats (Welford's / Chan's algorithm)
            delta = batch_mean - mean
            new_n = n + m
            mean = mean + delta * (m / new_n)
            M2 = M2 + batch_M2 + delta.pow(2) * (n * m / new_n)
            n = new_n

        # Finalize Variance and Std
        var = M2 / max(n - 1, 1)
        std = var.sqrt().clamp_min(1e-12)

        self._mean = mean.to(self.device)
        self._std = std.to(self.device)
        print("Normalization stats computed.")

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Override base normalization to use the precomputed global stats."""
        if self._mean is not None and self._std is not None:
            # Broadcasting works for (B, D) and (B, T, D) because mean/std are (D,)
            return (features - self._mean) / self._std
        return features

    def compute_raw(self, extractor, train_dataloader, val_dataloader=None, test_dataloader=None):
        # 1. Run the pre-computation pass
        self._precompute_normalization_stats(extractor, train_dataloader)

        # 2. Run the standard training loop
        return self.train_and_evaluate(extractor, train_dataloader, val_dataloader, test_dataloader)


class OnlineLinearRegressor(OnlineMetric):
    def __init__(
        self,
        input_feature_dim: int,
        num_classes: int = 0,
        output_dim: int = 1,
        lr_options: Optional[List[float]] = None,
        wd_options: Optional[List[float]] = None,
        n_epochs: int = 25,
        patience: int = 5,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        ceiling: Optional[np.ndarray] = None,
        # 'deterministic_scheduler', 'warmup_stable_decay', # Option for scheduler
        scheduler_type: str = 'reduce_on_plateau',
    ):
        """
        Online Linear Ridge Regressor with L2 regularization via SGD weight decay.

        Args:
            input_feature_dim: Dimensionality of input features
            num_classes: Unused for regression, kept for compatibility
            output_dim: Number of target variables for regression
            alpha: Ridge regularization strength (L2 penalty coefficient).
                   This is converted to weight_decay for the SGD optimizer.
            lr_options: Learning rates to try in grid search.
            wd_options: If provided, overrides the default weight_decay (2*alpha).
            n_epochs: Maximum number of training epochs.
            patience: Early stopping patience.
            batch_size: Batch size for training.
            device: Device to use ('cuda' or 'cpu').
            ceiling: Optional ceiling values for normalized metrics.
            scheduler_type: Type of LR scheduler to use. Options:
                            'reduce_on_plateau' (default): Adaptive LR based on validation score.
                            'warmup_stable_decay': Pre-defined schedule with warmup, stable, and decay phases.
        """
        self.alpha = [0, 0.05, 0.1]  # , 0.2, 0.25, 0.3, 0.35, 0.1]
        self._mean, self._std = None, None
        # Validate and store the scheduler type
        if scheduler_type not in ['reduce_on_plateau', 'warmup_stable_decay', 'deterministic_scheduler']:
            raise ValueError(
                f"Invalid scheduler_type: {scheduler_type}. Choose 'reduce_on_plateau' or 'warmup_stable_decay'.")
        self.scheduler_type = scheduler_type

        effective_wd_options = wd_options if wd_options else [
            2 * alpha for alpha in self.alpha]

        self.internal_model_type = "linear"

        super().__init__(
            num_classes=output_dim,
            input_feature_dim=input_feature_dim,
            internal_model_type="linear",
            lr_options=lr_options if lr_options else [1e-3, 1e-4],
            wd_options=effective_wd_options,
            n_epochs=n_epochs,
            patience=patience,
            batch_size=batch_size,
            device=device,
            ceiling=ceiling,
            task_type="regression",
        )
        self.internal_output_dim = output_dim

    @torch.no_grad()
    def _precompute_normalization_stats(self, extractor, dataloader):
        """
        Computes mean and std for feature normalization in a single pass.
        """
        n = 0
        mean = None
        M2 = None

        for batch_data, _ in tqdm(dataloader, desc="Pre-computing normalization stats"):
            feats = (extractor.extract_features_for_batch(batch_data)
                     if isinstance(batch_data, dict)
                     else extractor.extract_features_for_batch(batch_data.to(self.device)))
            X = feats.reshape(-1, feats.shape[-1])  # (M, D), where M=B*T

            # Per-batch stats
            m = X.shape[0]
            batch_mean = X.mean(dim=0)
            # Sum of squared deviations (M2) inside the batch
            batch_M2 = ((X - batch_mean) ** 2).sum(dim=0)

            if mean is None:
                mean = batch_mean.clone()
                M2 = batch_M2.clone()
                n = m
                continue

            # Merge running stats with batch stats (Chan et al. / parallel Welford)
            delta = batch_mean - mean
            new_n = n + m
            mean = mean + delta * (m / new_n)
            M2 = M2 + batch_M2 + delta.pow(2) * (n * m / new_n)
            n = new_n

        var = M2 / max(n - 1, 1)
        std = var.sqrt().clamp_min(1e-12)
        self._mean, self._std = mean, std

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        if self._mean is not None and self._std is not None:
            return (features - self._mean) / self._std
        return features

    def train_and_evaluate(
        self,
        extractor: 'OnlineFeatureExtractor',
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        scheduler_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train and evaluate Ridge regression model with a choice of LR scheduler.

        Args:
            extractor: The feature extractor model.
            train_dataloader: DataLoader for the training set.
            val_dataloader: DataLoader for the validation set.
            test_dataloader: DataLoader for the test set.
            scheduler_type: If provided, overrides the default scheduler for this run.

        Returns:
            A dictionary containing final evaluation scores and model details.
        """
        active_scheduler_type = scheduler_type if scheduler_type is not None else self.scheduler_type

        data_sample, labels_tensor = next(iter(train_dataloader))
        if isinstance(labels_tensor, tuple):
            labels_tensor = labels_tensor[1] if len(
                labels_tensor) > 1 else labels_tensor[0]
        self.internal_output_dim = labels_tensor.shape[-1] if labels_tensor.ndim > 1 else 1

        current_train_loader = train_dataloader
        val_loader_internal = val_dataloader

        if self.internal_model_type != "attention":
            self._precompute_normalization_stats(
                extractor, current_train_loader)

        param_grid = list(itertools.product(self.lr_options, self.wd_options))

        grid_search_bar = tqdm(param_grid, desc="Grid Search Hyperparameters")
        for lr, wd in grid_search_bar:
            grid_search_bar.set_postfix({"lr": lr, "wd": wd})

            internal_model = self._get_internal_model()
            internal_model = nn.DataParallel(internal_model)
            optimizer = optim.SGD(
                internal_model.parameters(),
                lr=lr,
                weight_decay=wd,
                momentum=0  # To make your SGD process as close as possible to the optimization problem that Ridge regression solves
            )

            if active_scheduler_type == 'reduce_on_plateau':
                scheduler = ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.1, patience=5)
            elif active_scheduler_type == 'deterministic_scheduler':
                total_steps = self.n_epochs * len(train_dataloader)
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
                )
            elif active_scheduler_type == 'warmup_stable_decay':
                total_steps = self.n_epochs * len(current_train_loader)
                warmup_steps = len(current_train_loader)
                decay_steps = len(current_train_loader) * 5
                stable_steps = total_steps - warmup_steps - decay_steps
                if stable_steps < 0:
                    raise ValueError(
                        "Total steps are too small for the Warmup-Stable-Decay schedule.")
                warmup_sched = LinearLR(optimizer, start_factor=min(
                    1e-6, 1e-2 / (lr * 1000)), end_factor=1.0, total_iters=warmup_steps)
                stable_sched = ConstantLR(
                    optimizer, factor=1.0, total_iters=stable_steps)
                decay_sched = LinearLR(
                    optimizer, start_factor=1.0, end_factor=0.0, total_iters=decay_steps)
                scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, stable_sched, decay_sched], milestones=[
                                         warmup_steps, warmup_steps + stable_steps])
            else:
                raise ValueError(
                    f"Unknown scheduler_type: {active_scheduler_type}")

            scaler = torch.amp.GradScaler(
                'cuda') if self.use_mixed_precision else None
            current_best_val_epoch_score = (-1) * float('inf')
            current_best_model_state_epoch = None
            epochs_no_improve = 0

            epoch_bar = tqdm(range(
                self.n_epochs), desc=f"Training (lr={lr}, wd={wd}, alpha={self.alpha})", leave=False)
            for epoch in epoch_bar:
                internal_model.train()
                total_train_loss = 0.0
                train_preds, train_true = [], []

                batch_bar = tqdm(
                    current_train_loader, desc=f"Epoch {epoch+1}/{self.n_epochs}", leave=False)
                for batch_idx, (batch_data, batch_labels_raw) in enumerate(batch_bar):
                    optimizer.zero_grad()
                    labels = self._unpack_labels(batch_labels_raw)

                    if self.use_mixed_precision:
                        with torch.amp.autocast('cuda'):
                            if isinstance(batch_data, dict):
                                features = extractor.extract_features_for_batch(
                                    batch_data)
                            else:
                                features = extractor.extract_features_for_batch(
                                    batch_data.to(self.device))
                            features = self._normalize_features(features)
                            outputs = internal_model(features)
                            outputs = torch.clamp(outputs, -10.0, +10.0)
                            loss = nn.functional.mse_loss(outputs, labels)
                            loss = self._stabilize_loss(loss)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            internal_model.parameters(), max_norm=self.gradient_clip_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if isinstance(batch_data, dict):
                            features = extractor.extract_features_for_batch(
                                batch_data)
                        else:
                            features = extractor.extract_features_for_batch(
                                batch_data.to(self.device))
                        features = self._normalize_features(features)
                        outputs = internal_model(features)
                        outputs = torch.clamp(outputs, -10.0, +10.0)
                        loss = nn.functional.mse_loss(outputs, labels)
                        loss = self._stabilize_loss(loss)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            internal_model.parameters(), max_norm=self.gradient_clip_norm)
                        optimizer.step()

                    if not active_scheduler_type == 'reduce_on_plateau':
                        scheduler.step()

                    total_train_loss += loss.item()
                    train_preds.append(outputs.detach().cpu())
                    train_true.append(labels.detach().cpu())
                    batch_mse = nn.functional.mse_loss(outputs, labels).item()
                    batch_bar.set_postfix(
                        {"batch_mse": f"{batch_mse:.4f}", "loss": f"{loss.item():.4f}"})

                    if wandb.run:
                        wandb.log({
                            "train/loss":   loss.item(),
                            "train/mse": batch_mse,
                            "train/lr":            scheduler.get_last_lr()[0],
                            "train/epoch":         epoch,
                            "train/batch_idx":     batch_idx,
                            "train/grid_lr":       lr,
                            "train/grid_wd":       wd,
                        }, step=getattr(self, "global_step", 0))
                        self.global_step = getattr(self, "global_step", 0) + 1

                avg_train_loss = total_train_loss / len(current_train_loader)
                train_preds_all = torch.cat(train_preds)
                train_true_all = torch.cat(train_true)
                train_mse = mean_squared_error(
                    train_true_all.numpy(), train_preds_all.numpy())
                train_corr = self._calculate_validation_score(
                    train_true_all, train_preds_all)

                if wandb.run:
                    wandb.log({
                        "train/median_pearson": train_corr,
                    }, step=getattr(self, "global_step", 0))

                internal_model.eval()

                val_preds, val_true = [], []
                with torch.no_grad():
                    val_bar = tqdm(
                        val_loader_internal, desc=f"Epoch {epoch+1} Validation", leave=False)
                    for val_batch_data, val_batch_labels_raw in val_bar:
                        labels = self._unpack_labels(val_batch_labels_raw)
                        if self.use_mixed_precision:
                            with torch.amp.autocast('cuda'):
                                if isinstance(val_batch_data, dict):
                                    features = extractor.extract_features_for_batch(
                                        val_batch_data)
                                else:
                                    features = extractor.extract_features_for_batch(
                                        val_batch_data.to(self.device))
                                features = self._normalize_features(features)
                                outputs = internal_model(features)
                        else:
                            if isinstance(val_batch_data, dict):
                                features = extractor.extract_features_for_batch(
                                    val_batch_data)
                            else:
                                features = extractor.extract_features_for_batch(
                                    val_batch_data.to(self.device))
                            features = self._normalize_features(features)
                            outputs = internal_model(features)
                        val_true.append(labels.cpu())
                        val_preds.append(outputs.cpu())

                val_preds_all = torch.cat(val_preds)
                val_true_all = torch.cat(val_true)
                val_mse = nn.functional.mse_loss(
                    val_preds_all, val_true_all).item()
                val_score = self._calculate_validation_score(
                    val_true_all, val_preds_all)

                if active_scheduler_type == 'reduce_on_plateau':
                    scheduler.step(val_score)

                epoch_bar.set_postfix({"train_mse": f"{train_mse:.5f}", "val_mse": f"{val_mse:.5f}",
                                      "train_corr": f"{train_corr:.5f}", "val_corr": f"{val_score:.5f}"})

                if wandb.run:
                    wandb.log({
                        "val/mse": val_mse,
                    }, step=getattr(self, "global_step", 0))

                if val_score > current_best_val_epoch_score:
                    current_best_val_epoch_score = val_score
                    current_best_model_state_epoch = internal_model.module.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    break

            if current_best_val_epoch_score > self.best_val_score:
                self.best_val_score = current_best_val_epoch_score
                self.best_internal_model_state = current_best_model_state_epoch
                self.best_hyperparams = {
                    "lr": lr, "wd": wd, "alpha": self.alpha, "scheduler": active_scheduler_type}

        grid_search_bar.close()

        final_scores = {}
        if test_dataloader and self.best_internal_model_state:
            best_model = self._get_internal_model()
            best_model.load_state_dict(self.best_internal_model_state)
            best_model = nn.DataParallel(best_model)
            best_model.eval()

            if self.feature_normalizer is not None:
                self.feature_normalizer.eval()

            test_preds_all, test_true_all = [], []
            with torch.no_grad():
                test_batch_bar = tqdm(
                    test_dataloader, desc="Evaluating on Test Set", leave=False)
                for test_batch_data, test_batch_labels_raw in test_batch_bar:
                    labels = self._unpack_labels(test_batch_labels_raw)
                    if self.use_mixed_precision:
                        with torch.amp.autocast('cuda'):
                            if isinstance(test_batch_data, dict):
                                features = extractor.extract_features_for_batch(
                                    test_batch_data)
                            else:
                                features = extractor.extract_features_for_batch(
                                    test_batch_data.to(self.device))
                            features = self._normalize_features(features)
                            outputs = best_model(features)
                    else:
                        if isinstance(test_batch_data, dict):
                            features = extractor.extract_features_for_batch(
                                test_batch_data)
                        else:
                            features = extractor.extract_features_for_batch(
                                test_batch_data.to(self.device))
                        features = self._normalize_features(features)
                        outputs = best_model(features)
                    test_preds_all.append(outputs.cpu())
                    test_true_all.append(labels.cpu())

            test_preds_all = torch.cat(test_preds_all)
            test_true_all = torch.cat(test_true_all)
            final_scores.update(self._calculate_final_scores(
                test_true_all, test_preds_all))
            final_scores['preds'] = test_preds_all.numpy()
            final_scores['gt'] = test_true_all.numpy()

        final_scores['best_hyperparams'] = self.best_hyperparams
        final_scores['best_val_score_during_grid_search'] = self.best_val_score
        return final_scores

    def compute_raw(
        self,
        extractor: 'OnlineFeatureExtractor',
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """
        Compute raw scores using Ridge regression.
        """
        return self.train_and_evaluate(
            extractor, train_dataloader, val_dataloader, test_dataloader
        )


class OnlineAttentionRegressor(OnlineLinearRegressor):
    def __init__(
        self,
        input_feature_dim: int,
        num_classes: int = 0,
        output_dim: int = 1,
        lr_options: Optional[List[float]] = None,
        wd_options: Optional[List[float]] = None,
        n_epochs: int = 25,
        patience: int = 5,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        ceiling: Optional[np.ndarray] = None,
        scheduler_type: str = 'reduce_on_plateau',
    ):
        """
        Online Attention-based Regressor with L2 regularization via SGD weight decay.
        Uses attention pooling instead of simple linear projection.

        Args:
            input_feature_dim: Dimensionality of input features
            num_classes: Unused for regression, kept for compatibility
            output_dim: Number of target variables for regression
            lr_options: Learning rates to try in grid search.
            wd_options: Weight decay options for grid search.
            n_epochs: Maximum number of training epochs.
            patience: Early stopping patience.
            batch_size: Batch size for training.
            device: Device to use ('cuda' or 'cpu').
            ceiling: Optional ceiling values for normalized metrics.
            scheduler_type: Type of LR scheduler to use.
        """
        # Set up alpha and validate scheduler type (copied from parent)
        self.alpha = [0, 0.05, 0.1]
        self._mean, self._std = None, None

        if scheduler_type not in ['reduce_on_plateau', 'warmup_stable_decay', 'deterministic_scheduler']:
            raise ValueError(
                f"Invalid scheduler_type: {scheduler_type}. Choose 'reduce_on_plateau' or 'warmup_stable_decay'.")
        self.scheduler_type = scheduler_type

        effective_wd_options = wd_options if wd_options else [
            2 * alpha for alpha in self.alpha]

        self.internal_model_type = "attention"
        # Call OnlineMetric.__init__ directly, skipping OnlineLinearRegressor's __init__
        OnlineMetric.__init__(
            self,
            num_classes=output_dim,
            input_feature_dim=input_feature_dim,
            internal_model_type="attention",  # Use attention instead of linear
            lr_options=lr_options if lr_options else [1e-4, 1e-5],
            wd_options=effective_wd_options,
            n_epochs=n_epochs,
            patience=patience,
            batch_size=batch_size,
            device=device,
            ceiling=ceiling,
            task_type="regression",
        )
        self.internal_output_dim = output_dim
