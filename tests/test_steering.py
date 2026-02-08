"""Tests for activation_steering.steering module."""

import pytest
import torch
import torch.nn as nn

from activation_steering.vectors import SteeringVector
from activation_steering.steering import (
    SteeringConfig,
    SteeringResult,
    ActivationSteerer,
)


class SimpleTransformerBlock(nn.Module):
    """Simple block for testing."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.linear(x)


class SimpleTransformer(nn.Module):
    """Simple model for testing."""
    def __init__(self, vocab_size=1000, hidden_dim=64, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.output(x)

    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        """Simple greedy generation for testing."""
        current = input_ids
        for _ in range(max_new_tokens):
            logits = self(current)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current = torch.cat([current, next_token], dim=1)
        return current


class SimpleTokenizer:
    """Simple tokenizer for testing."""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 0

    def __call__(self, text, return_tensors=None, padding=False):
        # Simple tokenization: just hash characters
        ids = [hash(c) % self.vocab_size for c in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        # Simple decode: just return a string representation
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(chr(65 + (i % 26)) for i in ids)


class TestSteeringConfig:
    def test_defaults(self):
        config = SteeringConfig()
        assert config.coefficient == 1.0
        assert config.position == "all"
        assert config.normalize is True

    def test_custom_config(self):
        config = SteeringConfig(
            coefficient=2.5,
            position="last",
            layers=[5, 10],
        )
        assert config.coefficient == 2.5
        assert config.layers == [5, 10]


class TestSteeringResult:
    def test_create(self):
        result = SteeringResult(
            text="Generated text",
            original_text="Original text",
            steering_applied=True,
        )
        assert result.text == "Generated text"
        assert result.steering_applied is True

    def test_to_dict(self):
        config = SteeringConfig(coefficient=1.5)
        result = SteeringResult(
            text="output",
            steering_applied=True,
            config=config,
            metadata={"key": "value"}
        )
        d = result.to_dict()
        assert d["text"] == "output"
        assert d["steering_applied"] is True
        assert d["config"]["coefficient"] == 1.5


class TestActivationSteerer:
    @pytest.fixture
    def model_and_tokenizer(self):
        model = SimpleTransformer(vocab_size=100, hidden_dim=64, num_layers=4)
        tokenizer = SimpleTokenizer(vocab_size=100)
        return model, tokenizer

    @pytest.fixture
    def steerer(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        return ActivationSteerer(model, tokenizer)

    @pytest.fixture
    def steering_vector(self):
        return SteeringVector(
            vector=torch.randn(64),
            layer_idx=2,
            name="test_vector",
        )

    def test_create_steerer(self, steerer):
        assert steerer.model is not None
        assert steerer.tokenizer is not None

    def test_steering_context(self, steerer, steering_vector):
        x = torch.randint(0, 100, (1, 10))

        with steerer.steering_context(steering_vector):
            # Hooks should be active
            assert len(steerer._active_hooks) > 0
            with torch.no_grad():
                out = steerer.model(x)

        # Hooks should be removed
        assert len(steerer._active_hooks) == 0

    def test_steering_context_multiple_vectors(self, steerer):
        vectors = [
            SteeringVector(torch.randn(64), layer_idx=1, name="v1"),
            SteeringVector(torch.randn(64), layer_idx=2, name="v2"),
        ]

        with steerer.steering_context(vectors):
            assert len(steerer._active_hooks) == 2

        assert len(steerer._active_hooks) == 0

    def test_generate_without_steering(self, steerer):
        result = steerer.generate(
            prompt="Hello",
            vectors=None,
            max_new_tokens=5,
        )
        assert isinstance(result, SteeringResult)
        assert result.steering_applied is False
        assert len(result.text) > 0

    def test_generate_with_steering(self, steerer, steering_vector):
        result = steerer.generate(
            prompt="Hello",
            vectors=steering_vector,
            max_new_tokens=5,
        )
        assert result.steering_applied is True
        assert result.config is not None

    def test_generate_with_compare(self, steerer, steering_vector):
        result = steerer.generate(
            prompt="Hello",
            vectors=steering_vector,
            max_new_tokens=5,
            compare_unsteered=True,
        )
        assert result.original_text is not None
        assert result.text is not None

    def test_sweep_coefficients(self, steerer, steering_vector):
        coefficients = [0.0, 0.5, 1.0, 2.0]
        results = steerer.sweep_coefficients(
            prompt="Test",
            vector=steering_vector,
            coefficients=coefficients,
            max_new_tokens=3,
        )
        assert len(results) == len(coefficients)
        for i, result in enumerate(results):
            assert result.metadata["coefficient"] == coefficients[i]

    def test_get_activations(self, steerer, steering_vector):
        acts = steerer.get_activations(
            text="Hello world",
            layer_idx=2,
        )
        assert acts.shape[0] == 1  # Batch
        assert acts.shape[-1] == 64  # Hidden dim

    def test_get_activations_with_steering(self, steerer, steering_vector):
        acts = steerer.get_activations(
            text="Hello world",
            layer_idx=steering_vector.layer_idx,
            vectors=steering_vector,
        )
        assert acts is not None

    def test_project_onto_vector(self, steerer, steering_vector):
        proj = steerer.project_onto_vector(
            text="Test text",
            vector=steering_vector,
        )
        # Should have shape (batch, seq)
        assert len(proj.shape) == 2


class TestSteeringConfigLayers:
    def test_apply_to_multiple_layers(self):
        model = SimpleTransformer(vocab_size=100, hidden_dim=64, num_layers=4)
        tokenizer = SimpleTokenizer(vocab_size=100)
        steerer = ActivationSteerer(model, tokenizer)

        vector = SteeringVector(torch.randn(64), layer_idx=1, name="multi")
        config = SteeringConfig(layers=[1, 2, 3])

        with steerer.steering_context(vector, config):
            # Should have hooks at layers 1, 2, 3
            assert len(steerer._active_hooks) == 3
