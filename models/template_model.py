import os


class TemplateModel:
    """Loads pre-trained models."""

    def __init__(self):
        """Initializes the Model loader."""
        # Static flag if this is image-model
        self.static = True

    def preprocess_fn(self, input_data):
        """
        Preprocesses input data for the model.

        Args:
            input_data: PIL Image, file path (str), or numpy array.

        Returns:
            torch.Tensor: Preprocessed input tensor.

        Raises:
            ValueError: If the input type is invalid.
        """
        pass

    def get_model(self, identifier):
        """
        Loads a model based on the identifier.

        Args:
            identifier (str): Identifier for the model variant.

        Returns:
            The loaded model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        pass

    def postprocess_fn(self, features_np):
        """Postprocesses model output by flattening features.

        Args:
            features_np (np.ndarray): Output features from model
                as a numpy array.

        Returns:
            np.ndarray: postprocessed feature tensor.
        """
        pass
