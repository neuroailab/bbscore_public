import os
from collections import deque
from typing import Any, Deque, Dict, List

import numpy as np
import torch
import torch.nn as nn

from sklearn.datasets import get_data_home
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPT2:
    """Loads pre-trained gpt2 model."""

    def __init__(self):
        """Initializes the loader."""
        self.model_mappings = {
            "GPT2-Small": ("gpt2"),
            "GPT2-Medium": ("gpt2-medium"),
            "GPT2-Large": ("gpt2-large"),
            "GPT2-XL": ("gpt2-xl")
        }

        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        # GPT-2 does not define a padding token by default; reuse EOS for padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
        self.context_length = 1024  # gpt2 context length
        # Current processor
        self.static = False
        self._pending_alignment: Deque[Dict[str, Any]] = deque()

    @staticmethod
    def _strip_token(token: str) -> str:
        if token.startswith("Ġ"):
            return token[1:]
        if token.startswith("##"):
            return token[2:]
        return token

    def _align_tokens_to_words(
        self,
        token_embeddings: torch.Tensor,
        tokens: List[str],
        words: List[str],
    ) -> torch.Tensor:
        stripped_tokens = [self._strip_token(token) for token in tokens]

        aligned_embeddings: List[torch.Tensor] = []
        token_index = 0

        for word in words:
            if word == "":
                continue

            current_pieces: List[str] = []
            current_embeddings: List[torch.Tensor] = []

            while token_index < len(stripped_tokens) and "".join(current_pieces) != word:
                current_pieces.append(stripped_tokens[token_index])
                current_embeddings.append(token_embeddings[token_index])
                token_index += 1

            if current_embeddings:
                stacked = torch.stack(current_embeddings, dim=0)
                aligned_embeddings.append(stacked.mean(dim=0))
            else:
                # Fallback: use the closest available embedding when we fail to match
                fallback = token_embeddings[min(
                    token_index, token_embeddings.size(0) - 1)]
                aligned_embeddings.append(fallback)

        if not aligned_embeddings:
            return token_embeddings[-1:].clone()

        return torch.stack(aligned_embeddings, dim=0)

    def preprocess_fn(self, input_data: str):
        """
        Tokenize input text for the LLM.

        Args:
            input_data: text + its context (all words before up to context length)

        Returns:
            torch.Tensor: Token IDs padded/truncated to the model context length.

        Raises:
            ValueError: If processor is not initialized or input is invalid
        """

        # self.tokenizer(input_string, return_tensors='pt')['input_ids'][0]

        if torch.is_tensor(input_data):
            return input_data

        if isinstance(input_data, str):
            encoded = self.tokenizer(
                input_data,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.context_length,
            )
            input_ids = encoded['input_ids'].squeeze(0)
            words = input_data.split()
            attention_mask = encoded['attention_mask'].squeeze(0)
            valid_length = int(attention_mask.sum().item(
            )) if attention_mask.numel() > 0 else len(input_ids)
            valid_length = max(1, min(valid_length, input_ids.size(0)))
            trimmed_ids = input_ids[-valid_length:].clone()

            # Store alignment metadata to be consumed during post-processing
            self._pending_alignment.append(
                {
                    'input_ids': trimmed_ids,
                    'valid_length': valid_length,
                    'words': words,
                }
            )
            return input_ids

        if isinstance(input_data, list):
            if not input_data:
                return torch.empty((0, 0), dtype=torch.long)
            if not all(isinstance(elem, str) for elem in input_data):
                raise ValueError(
                    "All elements must be strings when providing a list of inputs.")

            encoded = self.tokenizer(
                input_data,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.context_length,
            )

            all_words = [text.split() for text in input_data]
            for idx, words in enumerate(all_words):
                attention_mask = encoded['attention_mask'][idx]
                valid_length = int(attention_mask.sum().item()) if attention_mask.numel(
                ) > 0 else encoded['input_ids'].size(1)
                valid_length = max(
                    1, min(valid_length, encoded['input_ids'].size(1)))
                trimmed_ids = encoded['input_ids'][idx, -valid_length:].clone()
                self._pending_alignment.append(
                    {
                        'input_ids': trimmed_ids,
                        'valid_length': valid_length,
                        'words': words,
                    }
                )

            return encoded['input_ids']

        raise ValueError(
            "Input should be a string, list of strings, or tensor of token IDs.")

    def get_model(self, identifier):
        """
        Loads a GPT2 model based on the identifier.

        Args:
            identifier (str):  Identifier for the GPT2 variant.

        Returns:
            model: The loaded GPT2 model.

        Raises:
            ValueError: If the identifier is unknown.
        """
        for prefix, model_params in self.model_mappings.items():
            if identifier == prefix:
                model = AutoModelForCausalLM.from_pretrained(model_params)
                return model

    def postprocess_fn(self, features):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)

        if features.dim() == 4 and features.size(1) == 1:
            features = features[:, 0]
        elif features.dim() == 4:
            features = features.view(features.size(0), features.size(
                1) * features.size(2), features.size(3))

        if features.dim() != 3:
            return features

        aligned_batch: List[torch.Tensor] = []

        for sample_idx in range(features.size(0)):
            if not self._pending_alignment:
                aligned_batch.append(features[sample_idx, -1])
                continue

            metadata = self._pending_alignment.popleft()
            words = metadata['words']
            valid_length = metadata.get('valid_length', features.size(1))
            valid_length = max(1, min(valid_length, features.size(1)))
            token_embeddings = features[sample_idx]
            token_embeddings = token_embeddings[-valid_length:]

            token_ids = metadata['input_ids']
            if token_ids.size(0) != valid_length:
                token_ids = token_ids[-valid_length:]

            token_list = self.tokenizer.convert_ids_to_tokens(
                token_ids.tolist())
            aligned_embeddings = self._align_tokens_to_words(
                token_embeddings,
                token_list,
                words,
            )

            # Represent the current stimulus with the last word embedding
            last_embedding = aligned_embeddings[-1]
            aligned_batch.append(last_embedding)

        return torch.stack(aligned_batch, dim=0)
