"""Tests for activation_steering.vectors module."""

import pytest
import torch
import tempfile
import os
from pathlib import Path

from activation_steering.vectors import (
    SteeringVector,
    ContrastivePair,
    save_steering_vector,
    load_steering_vector,
    HONESTY_PAIRS,
    HELPFULNESS_PAIRS,
)


class TestContrastivePair:
    def test_create(self):
        pair = ContrastivePair(
            positive="I'll help you",
            negative="I won't help",
            label="helpful",
        )
        assert pair.positive == "I'll help you"
        assert pair.negative == "I won't help"
        assert pair.label == "helpful"
        assert pair.weight == 1.0

    def test_to_dict(self):
        pair = ContrastivePair("pos", "neg", "label", 0.5)
        d = pair.to_dict()
        assert d["positive"] == "pos"
        assert d["negative"] == "neg"
        assert d["weight"] == 0.5

    def test_from_dict(self):
        data = {
            "positive": "yes",
            "negative": "no",
            "label": "test",
            "weight": 2.0,
        }
        pair = ContrastivePair.from_dict(data)
        assert pair.positive == "yes"
        assert pair.weight == 2.0


class TestSteeringVector:
    def test_create(self):
        vec = torch.randn(768)
        sv = SteeringVector(
            vector=vec,
            layer_idx=5,
            name="test_vector",
            description="A test vector",
        )
        assert sv.dim == 768
        assert sv.layer_idx == 5
        assert sv.name == "test_vector"

    def test_norm(self):
        vec = torch.ones(100)
        sv = SteeringVector(vec, layer_idx=0)
        assert abs(sv.norm - 10.0) < 0.01  # sqrt(100)

    def test_normalize(self):
        vec = torch.randn(768)
        sv = SteeringVector(vec, layer_idx=0)
        normalized = sv.normalize()
        assert abs(normalized.norm - 1.0) < 1e-5

    def test_scale(self):
        vec = torch.ones(100)
        sv = SteeringVector(vec, layer_idx=0)
        scaled = sv.scale(2.0)
        assert abs(scaled.norm - 20.0) < 0.01

    def test_to_device(self):
        vec = torch.randn(768)
        sv = SteeringVector(vec, layer_idx=0)
        moved = sv.to("cpu")
        assert moved.vector.device == torch.device("cpu")

    def test_project_onto(self):
        # Create a simple steering vector
        vec = torch.zeros(768)
        vec[0] = 1.0  # Unit vector in first dimension
        sv = SteeringVector(vec, layer_idx=0)

        # Create activations with known projection
        acts = torch.zeros(2, 10, 768)
        acts[..., 0] = 5.0  # Should project to 5.0

        proj = sv.project_onto(acts)
        assert proj.shape == (2, 10)
        assert torch.allclose(proj, torch.full((2, 10), 5.0), atol=1e-5)

    def test_add(self):
        v1 = SteeringVector(torch.ones(100), layer_idx=5, name="v1")
        v2 = SteeringVector(torch.ones(100) * 2, layer_idx=5, name="v2")
        combined = v1 + v2
        assert combined.layer_idx == 5
        assert torch.allclose(combined.vector, torch.ones(100) * 3)

    def test_add_different_layers_raises(self):
        v1 = SteeringVector(torch.ones(100), layer_idx=5)
        v2 = SteeringVector(torch.ones(100), layer_idx=10)
        with pytest.raises(AssertionError):
            _ = v1 + v2

    def test_neg(self):
        vec = torch.ones(100) * 3
        sv = SteeringVector(vec, layer_idx=0, name="pos")
        neg = -sv
        assert neg.name == "-pos"
        assert torch.allclose(neg.vector, torch.ones(100) * -3)

    def test_to_dict(self):
        vec = torch.randn(768)
        sv = SteeringVector(
            vec, layer_idx=5, name="test", description="desc",
            metadata={"source": "contrastive"}
        )
        d = sv.to_dict()
        assert d["name"] == "test"
        assert d["layer_idx"] == 5
        assert d["dim"] == 768
        assert "norm" in d


class TestSaveLoad:
    def test_save_and_load(self):
        vec = torch.randn(768)
        sv = SteeringVector(
            vec, layer_idx=5, name="save_test",
            description="Testing save/load",
            metadata={"key": "value"}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_vector"
            save_steering_vector(sv, path)

            # Check files exist
            assert path.with_suffix(".pt").exists()
            assert path.with_suffix(".json").exists()

            # Load and verify
            loaded = load_steering_vector(path)
            assert loaded.layer_idx == 5
            assert loaded.name == "save_test"
            assert torch.allclose(loaded.vector, vec)

    def test_load_with_device(self):
        vec = torch.randn(768)
        sv = SteeringVector(vec, layer_idx=0, name="device_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_vector"
            save_steering_vector(sv, path)

            loaded = load_steering_vector(path, device="cpu")
            assert loaded.vector.device == torch.device("cpu")


class TestPredefinedPairs:
    def test_honesty_pairs(self):
        assert len(HONESTY_PAIRS) >= 2
        for pair in HONESTY_PAIRS:
            assert isinstance(pair, ContrastivePair)
            assert pair.label == "honesty"

    def test_helpfulness_pairs(self):
        assert len(HELPFULNESS_PAIRS) >= 2
        for pair in HELPFULNESS_PAIRS:
            assert isinstance(pair, ContrastivePair)
            assert pair.label == "helpfulness"
