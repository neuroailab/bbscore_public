import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader  # Keep for potential dummy runs
from tqdm import tqdm
from typing import List, Sequence, Tuple, Any, Optional


class OnlineFeatureExtractor:
    """
    Wrapper for feature extraction from models, designed for online batch processing.
    """

    def __init__(
        self,
        model: nn.Module,  # The actual PyTorch model
        layer_name: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        postprocess_fn: Optional[callable] = None,
        static: bool = True,  # From model_instance.static
        sequence_mode: str = "all"  # "last", "all", "concatenate"
    ):
        self.device = device
        self.model = model  # This is the nn.Module
        self.layer_name = layer_name
        self.postprocess_fn = postprocess_fn
        self.static = static
        self.sequence_mode = sequence_mode
        self._has_printed_static_warning = False  # Flag to control printing

        if not any(name == layer_name for name, _ in self.model.named_modules()):
            raise ValueError(
                f"Layer {layer_name} not found in model {self.model.__class__.__name__}. "
                f"Available layers: {[n for n, _ in self.model.named_modules()]}. \n"
                f"Model: {self.model}"
            )

        self.model.eval()
        self.model.to(device)
        self.features_hook_output = []  # Stores output from the hook for the current batch
        self._register_hooks()

    def _hook_fn(self, module, input, output):
        self.features_hook_output.append(output)

    def _register_hooks(self):
        def get_module_by_name(model, layer_name):
            for name, module in model.named_modules():
                if name == layer_name:
                    return module
            raise ValueError(f"Layer {layer_name} not found in model")

        target_layer = get_module_by_name(self.model, self.layer_name)
        if isinstance(target_layer, torch.nn.ModuleList):
            print(
                f"Warning: Target layer '{self.layer_name}' is a ModuleList. Using the last module.")
            target_layer[-1].register_forward_hook(self._hook_fn)
        else:
            target_layer.register_forward_hook(self._hook_fn)

    def _get_activations_for_batch(self, inputs: torch.Tensor) -> None:
        """Passes inputs through the model. Features are captured by the hook."""
        self.features_hook_output = []  # Clear for current batch
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            self.model(inputs)  # This triggers the hook

    @classmethod
    def _set_weights(cls, outputs: List[torch.Tensor], shape: Sequence[int]) -> torch.Tensor:
        # (Copied from original FeatureExtractor, ensure it's appropriate for online use)
        rank = len(shape)
        if rank == 5:
            return torch.cat([out[: shape[0], None, : shape[1], : shape[2], : shape[3], : shape[4]] for out in outputs], dim=1)
        elif rank == 4:
            return torch.cat([out[: shape[0], None, : shape[1], : shape[2], : shape[3]] for out in outputs], dim=1)
        elif rank == 3:
            return torch.cat([out[: shape[0], None, : shape[1], : shape[2]] for out in outputs], dim=1)
        elif rank == 2:
            return torch.cat([out[: shape[0], None, : shape[1]] for out in outputs], dim=1)
        elif rank == 1:
            return torch.cat([out[: shape[0], None] for out in outputs], dim=1)
        else:
            raise ValueError(f"Unexpected target shape rank: {shape}")

    def _process_sequence_features(self, features_list: List[Any]) -> Optional[torch.Tensor]:
        """Process sequential features according to the specified sequence_mode."""
        if not features_list:
            return None

        processed_features = []
        for i, feat in enumerate(features_list):
            # Handle standard models that return tuples (output, hidden)
            if isinstance(feat, tuple):
                feat = feat[0]

            # CHECK FOR HMAX CASE: List/Tuple of tensors
            # We must preserve the list structure for the "all" block to handle it correctly.
            if isinstance(feat, (list, tuple)) and len(feat) > 0 and all(isinstance(t, torch.Tensor) for t in feat):
                # Move individual tensors to CPU immediately
                feat = [t.cpu() for t in feat]
            else:
                # Standard case: ensure it is a single tensor
                if not isinstance(feat, torch.Tensor):  # Ensure tensor
                    feat = torch.tensor(feat)
                # Move to CPU immediately to free GPU memory for long sequences
                feat = feat.cpu()

            processed_features.append(feat)

            # Clear the GPU tensor reference immediately
            features_list[i] = None

        # Clear the entire features_hook_output to free GPU memory
        self.features_hook_output.clear()

        if self.sequence_mode == "last":
            # If the last feature is an HMAX list, this might fail or need handling,
            # but usually HMAX is used with 'all' or implies static 'all'.
            # Assuming standard behavior for 'last':
            res = processed_features[-1]
            if isinstance(res, list):
                # Fallback if user requests 'last' on HMAX: treat as 'all' for that frame
                nested = res
                min_shape = [min(t.shape[d] for t in nested)
                             for d in range(nested[0].ndim)]
                result = self._set_weights(nested, min_shape)
                return result.to(self.device)
            return res.to(self.device)

        elif self.sequence_mode == "all":
            # HMAX specific patch
            if (len(processed_features) == 1 and isinstance(processed_features[0], (list, tuple)) and
                    all(isinstance(t, torch.Tensor) for t in processed_features[0])):
                nested = processed_features[0]
                min_shape = [min(t.shape[d] for t in nested)
                             for d in range(nested[0].ndim)]
                result = self._set_weights(nested, min_shape)
                return result.to(self.device)

            elif all(isinstance(f, torch.Tensor) for f in processed_features):
                try:
                    # Stacks along new dim: (B, T_hook, D...)
                    result = torch.stack(processed_features, dim=1)
                    return result.to(self.device)
                except RuntimeError as e:
                    print(
                        f"Error stacking features in 'all' mode: {e}. Feature shapes: {[f.shape for f in processed_features]}")
                    # Fallback: concatenate along feature dimension if stacking fails due to varying T_hook
                    flat_features = [f.reshape(f.shape[0], -1)
                                     for f in processed_features]
                    result = torch.cat(flat_features, dim=1)
                    return result.to(self.device)
            else:
                result = torch.tensor(np.array(processed_features))
                return result.to(self.device)
        elif self.sequence_mode == "concatenate":
            if all(isinstance(f, torch.Tensor) for f in processed_features):
                flat_features = [f.reshape(f.shape[0], -1)
                                 for f in processed_features]
                result = torch.cat(flat_features, dim=1)
                return result.to(self.device)
            else:
                # Handle case where features might be lists (HMAX) but user requested concat
                # Flatten the lists if they exist
                flat_list = []
                for f in processed_features:
                    if isinstance(f, list):
                        # If HMAX list, we likely need to align them first or just flatten everything?
                        # Assuming simple flatten for 'concatenate' mode
                        for sub_f in f:
                            flat_list.append(torch.tensor(
                                np.array(sub_f).reshape(sub_f.shape[0], -1)))
                    else:
                        flat_list.append(torch.tensor(
                            np.array(f).reshape(f.shape[0], -1)))

                result = torch.cat(flat_list, dim=1)
                return result.to(self.device)
        else:
            raise ValueError(f"Unknown sequence_mode: {self.sequence_mode}")

    def extract_features_for_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Extracts features for a single batch of inputs.
        Args:
            inputs: A batch of input data (e.g., from DataLoader), already preprocessed.
        Returns:
            torch.Tensor: Extracted features for the batch.
        """
        B_original, T_original = None, None

        # (B, T, C, H, W) for static model
        if self.static and inputs.ndim >= 5:
            B_original, T_original = inputs.shape[0], inputs.shape[1]
            # (B*T, C, H, W)
            inputs_reshaped = inputs.reshape(-1, *inputs.shape[2:])
            self._get_activations_for_batch(inputs_reshaped)
        else:  # (B, C, H, W) for static, or (B, C, T, H, W) for video model
            self._get_activations_for_batch(inputs)

        if not self.features_hook_output:
            raise RuntimeError("No features were captured by the hook.")

        # features_val will be a tensor after this
        features_val = self._process_sequence_features(
            self.features_hook_output)

        if features_val is None:
            raise RuntimeError("Feature processing resulted in None.")

        # Reshape back if static model processed video input frame by frame
        if self.static and B_original is not None:
            if self.sequence_mode == "all":
                if features_val.ndim == 3 and features_val.shape[1] > 1:
                    # Check flag so we only print once
                    if not self._has_printed_static_warning:
                        print(
                            "We have a static model with T_hook > 1. Averaging over T_hook dim:", features_val.shape)
                        self._has_printed_static_warning = True
                    features_val = features_val.mean(dim=1)  # (B*T, D_sub)
                features_val = features_val.reshape(B_original, T_original, -1)
            else:  # "last" or "concatenate", output is (B*T, D_feat)
                features_val = features_val.reshape(B_original, T_original, -1)

        if self.postprocess_fn:
            # Handle both numpy and tensor postprocessing functions
            try:
                # Try tensor input first
                processed = self.postprocess_fn(features_val)
                if isinstance(processed, torch.Tensor):
                    features_val = processed
                else:
                    # If returns numpy, convert back to tensor
                    features_val = torch.from_numpy(processed).to(self.device)
            except (TypeError, AttributeError):
                # Fallback to numpy input (most common case)
                features_np = features_val.cpu().numpy()
                processed = self.postprocess_fn(features_np)
                if isinstance(processed, np.ndarray):
                    features_val = torch.from_numpy(processed).to(self.device)
                else:
                    features_val = processed.to(self.device)

        return features_val.detach()  # Ensure it's detached for the metric model

    def get_feature_dimensionality(self, dummy_input_batch: torch.Tensor) -> int:
        """
        Determines the output feature dimensionality by passing a dummy batch.
        Args:
            dummy_input_batch: A tensor representative of a single batch of preprocessed input.
        Returns:
            int: The dimensionality of the features produced for one item in a batch
                 (after sequence processing and postprocessing).
        """
        print(
            f"Determining feature dimensionality with dummy batch")
        features = self.extract_features_for_batch(dummy_input_batch)
        print(features.shape)
        # features shape can be (B, D) or (B, T, D)
        if features.ndim == 2:  # (B, D)
            shape = features.shape[1]
        elif features.ndim == 3:  # (B, T, D)
            shape = features.shape[-1]
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        features = None
        return shape


class IdentityFeatureExtractor:
    """
    A dummy extractor for Assembly benchmarks where the input
    is already the feature vector (e.g. neural data).
    """

    def __init__(self, device='cuda'):
        self.device = device
        # A dummy model to satisfy attribute checks if necessary
        self.model = nn.Identity()
        self.sequence_mode = "all"  # Pass through as-is

    def extract_features_for_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        # Simply move to device and return
        if isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        return torch.tensor(inputs).to(self.device)

    def get_feature_dimensionality(self, dummy_input: torch.Tensor) -> int:
        # Return the last dimension size
        if dummy_input.ndim == 2:
            return dummy_input.shape[1]
        elif dummy_input.ndim == 3:
            return dummy_input.shape[2]
        return dummy_input.shape[-1]
