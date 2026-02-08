"""Tests for activation_steering.analysis module."""

import pytest
import torch
import torch.nn as nn

from activation_steering.vectors import SteeringVector, ContrastivePair
from activation_steering.analysis import (
    ProjectionAnalysis,
    SteeringEffectiveness,
    ActivationAnalyzer,
)


class SimpleTransformerBlock(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.linear(x)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=64, num_layers=4):
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


class SimpleTokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def __call__(self, text, return_tensors=None, padding=False):
        ids = [hash(c) % self.vocab_size for c in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}


class TestProjectionAnalysis:
    def test_create(self):
        proj = torch.randn(100)
        analysis = ProjectionAnalysis(
            projections=proj,
            mean_projection=proj.mean().item(),
            std_projection=proj.std().item(),
            min_projection=proj.min().item(),
            max_projection=proj.max().item(),
            layer_idx=5,
        )
        assert analysis.layer_idx == 5
        assert abs(analysis.mean_projection - proj.mean().item()) < 1e-5

    def test_to_dict(self):
        analysis = ProjectionAnalysis(
            projections=torch.randn(10),
            mean_projection=0.5,
            std_projection=0.1,
            min_projection=-0.5,
            max_projection=1.0,
            layer_idx=3,
        )
        d = analysis.to_dict()
        assert d["mean_projection"] == 0.5
        assert d["layer_idx"] == 3


class TestSteeringEffectiveness:
    def test_create(self):
        eff = SteeringEffectiveness(
            projection_shift=0.5,
            cosine_similarity=0.8,
            magnitude_ratio=0.1,
            layer_idx=5,
            coefficient=1.0,
        )
        assert eff.projection_shift == 0.5
        assert eff.cosine_similarity == 0.8

    def test_to_dict(self):
        eff = SteeringEffectiveness(
            projection_shift=0.3,
            cosine_similarity=0.9,
            magnitude_ratio=0.05,
            layer_idx=2,
            coefficient=1.5,
        )
        d = eff.to_dict()
        assert d["projection_shift"] == 0.3
        assert d["coefficient"] == 1.5


class TestActivationAnalyzer:
    @pytest.fixture
    def analyzer(self):
        model = SimpleTransformer(vocab_size=100, hidden_dim=64, num_layers=4)
        tokenizer = SimpleTokenizer(vocab_size=100)
        return ActivationAnalyzer(model, tokenizer)

    @pytest.fixture
    def steering_vector(self):
        return SteeringVector(
            vector=torch.randn(64),
            layer_idx=2,
            name="test",
        )

    def test_analyze_projection(self, analyzer, steering_vector):
        texts = ["Hello world", "Test text", "Another example"]
        analysis = analyzer.analyze_projection(texts, steering_vector)

        assert isinstance(analysis, ProjectionAnalysis)
        assert analysis.layer_idx == 2
        assert len(analysis.projections) > 0

    def test_compare_projections(self, analyzer, steering_vector):
        positive = ["I am happy", "Great news", "Wonderful"]
        negative = ["I am sad", "Bad news", "Terrible"]

        comparison = analyzer.compare_projections(positive, negative, steering_vector)

        assert "positive_mean" in comparison
        assert "negative_mean" in comparison
        assert "separation" in comparison
        assert "effect_size" in comparison

    def test_measure_steering_effect(self, analyzer, steering_vector):
        text = "This is a test sentence"
        effectiveness = analyzer.measure_steering_effect(
            text, steering_vector, coefficient=1.0
        )

        assert isinstance(effectiveness, SteeringEffectiveness)
        assert effectiveness.layer_idx == 2
        assert effectiveness.coefficient == 1.0

    def test_activation_statistics(self, analyzer):
        texts = ["Hello", "World", "Test"]
        stats = analyzer.activation_statistics(texts, layer_idx=1)

        assert "mean" in stats
        assert "std" in stats
        assert "global_mean" in stats
        assert stats["layer_idx"] == 1

    def test_vector_similarity(self, analyzer):
        v1 = SteeringVector(torch.randn(64), layer_idx=0, name="v1")
        v2 = SteeringVector(torch.randn(64), layer_idx=0, name="v2")

        sim = analyzer.vector_similarity(v1, v2)
        assert -1.0 <= sim <= 1.0

        # Same vector should have similarity 1.0
        same_sim = analyzer.vector_similarity(v1, v1)
        assert abs(same_sim - 1.0) < 1e-5

    def test_decompose_vector(self, analyzer):
        # Create orthogonal basis
        basis1 = SteeringVector(
            torch.cat([torch.ones(32), torch.zeros(32)]),
            layer_idx=0, name="basis1"
        )
        basis2 = SteeringVector(
            torch.cat([torch.zeros(32), torch.ones(32)]),
            layer_idx=0, name="basis2"
        )

        # Target vector that's a combination
        target = SteeringVector(
            torch.cat([torch.ones(32) * 2, torch.ones(32) * 3]),
            layer_idx=0, name="target"
        )

        coeffs = analyzer.decompose_vector(target, [basis1, basis2])

        assert "basis1" in coeffs
        assert "basis2" in coeffs
        assert "residual_norm" in coeffs


class TestVectorComparison:
    def test_similar_vectors(self):
        vec = torch.randn(64)
        v1 = SteeringVector(vec, layer_idx=0, name="v1")
        v2 = SteeringVector(vec + torch.randn(64) * 0.01, layer_idx=0, name="v2")

        model = SimpleTransformer(vocab_size=100, hidden_dim=64, num_layers=4)
        tokenizer = SimpleTokenizer(vocab_size=100)
        analyzer = ActivationAnalyzer(model, tokenizer)

        sim = analyzer.vector_similarity(v1, v2)
        # Should be very similar
        assert sim > 0.9

    def test_opposite_vectors(self):
        vec = torch.randn(64)
        v1 = SteeringVector(vec, layer_idx=0, name="v1")
        v2 = SteeringVector(-vec, layer_idx=0, name="v2")

        model = SimpleTransformer(vocab_size=100, hidden_dim=64, num_layers=4)
        tokenizer = SimpleTokenizer(vocab_size=100)
        analyzer = ActivationAnalyzer(model, tokenizer)

        sim = analyzer.vector_similarity(v1, v2)
        # Should be exactly opposite
        assert abs(sim + 1.0) < 1e-5
