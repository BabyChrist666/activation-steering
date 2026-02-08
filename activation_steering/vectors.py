"""
Steering vector extraction and management.

Provides tools for creating, saving, and loading steering vectors
that can be used to control model behavior.
"""

import torch
import torch.nn as nn
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path


@dataclass
class ContrastivePair:
    """
    A pair of texts representing contrasting behaviors.

    Used to extract steering vectors by computing the difference
    in activations between positive and negative examples.
    """
    positive: str
    negative: str
    label: Optional[str] = None
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {
            "positive": self.positive,
            "negative": self.negative,
            "label": self.label,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContrastivePair":
        return cls(
            positive=data["positive"],
            negative=data["negative"],
            label=data.get("label"),
            weight=data.get("weight", 1.0),
        )


@dataclass
class SteeringVector:
    """
    A learned vector that can steer model activations.

    The vector is extracted from contrastive pairs and can be
    added to activations at a specific layer to change behavior.
    """
    vector: torch.Tensor
    layer_idx: int
    name: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return self.vector.shape[-1]

    @property
    def norm(self) -> float:
        return self.vector.norm().item()

    def normalize(self) -> "SteeringVector":
        """Return a unit-norm version of the vector."""
        return SteeringVector(
            vector=self.vector / self.vector.norm(),
            layer_idx=self.layer_idx,
            name=self.name,
            description=self.description,
            metadata=self.metadata,
        )

    def scale(self, factor: float) -> "SteeringVector":
        """Scale the vector by a factor."""
        return SteeringVector(
            vector=self.vector * factor,
            layer_idx=self.layer_idx,
            name=self.name,
            description=self.description,
            metadata=self.metadata,
        )

    def to(self, device: Union[str, torch.device]) -> "SteeringVector":
        """Move vector to device."""
        return SteeringVector(
            vector=self.vector.to(device),
            layer_idx=self.layer_idx,
            name=self.name,
            description=self.description,
            metadata=self.metadata,
        )

    def project_onto(self, activations: torch.Tensor) -> torch.Tensor:
        """Project activations onto the steering direction."""
        vec_norm = self.vector / self.vector.norm()
        return torch.einsum("...d,d->...", activations, vec_norm)

    def __add__(self, other: "SteeringVector") -> "SteeringVector":
        assert self.layer_idx == other.layer_idx, "Layer indices must match"
        return SteeringVector(
            vector=self.vector + other.vector,
            layer_idx=self.layer_idx,
            name=f"{self.name}+{other.name}",
            description="Combined steering vector",
            metadata={"combined_from": [self.name, other.name]},
        )

    def __neg__(self) -> "SteeringVector":
        return SteeringVector(
            vector=-self.vector,
            layer_idx=self.layer_idx,
            name=f"-{self.name}",
            description=f"Negated: {self.description}",
            metadata=self.metadata,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "layer_idx": self.layer_idx,
            "dim": self.dim,
            "norm": self.norm,
            "metadata": self.metadata,
        }


def extract_steering_vector(
    model: nn.Module,
    tokenizer,
    pairs: List[ContrastivePair],
    layer_idx: int,
    hook_point: str = "residual",
    aggregation: str = "mean_diff",
    device: Optional[str] = None,
) -> SteeringVector:
    """
    Extract a steering vector from contrastive pairs.

    Args:
        model: The transformer model.
        tokenizer: Tokenizer for the model.
        pairs: List of contrastive text pairs.
        layer_idx: Layer to extract from.
        hook_point: Where to hook ("residual", "mlp", "attn").
        aggregation: How to aggregate ("mean_diff", "pca", "last_token").
        device: Device to use.

    Returns:
        Extracted steering vector.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    positive_activations = []
    negative_activations = []

    def get_hook_fn(storage: list):
        def hook_fn(module, inputs, outputs):
            # Handle different output formats
            if isinstance(outputs, tuple):
                act = outputs[0]
            else:
                act = outputs
            # Store last token activation (or mean across sequence)
            if aggregation == "last_token":
                storage.append(act[:, -1, :].detach().cpu())
            else:
                storage.append(act.mean(dim=1).detach().cpu())
        return hook_fn

    # Register hook at the specified layer
    if hasattr(model, "blocks"):
        block = model.blocks[layer_idx]
    elif hasattr(model, "transformer"):
        if hasattr(model.transformer, "h"):
            block = model.transformer.h[layer_idx]
        else:
            block = model.transformer.blocks[layer_idx]
    elif hasattr(model, "layers"):
        block = model.layers[layer_idx]
    else:
        raise ValueError("Could not find layer blocks in model")

    # Process positive examples
    for pair in pairs:
        positive_activations.clear()
        hook = block.register_forward_hook(get_hook_fn(positive_activations))

        try:
            inputs = tokenizer(pair.positive, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                model(**inputs)

            neg_activations_local = []
            hook.remove()
            hook = block.register_forward_hook(get_hook_fn(neg_activations_local))

            inputs = tokenizer(pair.negative, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                model(**inputs)

            negative_activations.append(neg_activations_local[0] * pair.weight)
            positive_activations_weighted = positive_activations[0] * pair.weight
            positive_activations.clear()
            positive_activations.append(positive_activations_weighted)

        finally:
            hook.remove()

        if len(positive_activations) > 0:
            positive_activations = [positive_activations[-1]]

    # Aggregate to get steering vector
    if aggregation == "mean_diff":
        pos_mean = torch.stack([p for p in positive_activations if len(p.shape) >= 1]).mean(dim=0)
        neg_mean = torch.stack([n for n in negative_activations if len(n.shape) >= 1]).mean(dim=0)
        vector = (pos_mean - neg_mean).squeeze()
    elif aggregation == "pca":
        # Use PCA on the difference vectors
        diffs = []
        for p, n in zip(positive_activations, negative_activations):
            diffs.append((p - n).squeeze())
        diff_stack = torch.stack(diffs)
        # First principal component
        _, _, V = torch.svd(diff_stack)
        vector = V[:, 0]
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return SteeringVector(
        vector=vector.to(device),
        layer_idx=layer_idx,
        name="steering_vector",
        description=f"Extracted from {len(pairs)} pairs at layer {layer_idx}",
        metadata={
            "num_pairs": len(pairs),
            "aggregation": aggregation,
            "hook_point": hook_point,
        },
    )


def save_steering_vector(
    vector: SteeringVector,
    path: Union[str, Path],
) -> None:
    """Save a steering vector to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save tensor
    torch.save(vector.vector, path.with_suffix(".pt"))

    # Save metadata
    meta = vector.to_dict()
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_steering_vector(
    path: Union[str, Path],
    device: Optional[str] = None,
) -> SteeringVector:
    """Load a steering vector from disk."""
    path = Path(path)

    # Load tensor
    tensor_path = path.with_suffix(".pt") if not str(path).endswith(".pt") else path
    vector = torch.load(tensor_path, map_location=device or "cpu")

    # Load metadata
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {"name": path.stem, "layer_idx": 0}

    return SteeringVector(
        vector=vector,
        layer_idx=meta.get("layer_idx", 0),
        name=meta.get("name", ""),
        description=meta.get("description", ""),
        metadata=meta.get("metadata", {}),
    )


# Pre-defined contrastive pairs for common steering directions
HONESTY_PAIRS = [
    ContrastivePair(
        positive="I honestly don't know the answer to that.",
        negative="I definitely know the answer, it's obvious.",
        label="honesty",
    ),
    ContrastivePair(
        positive="I might be wrong about this, but I think...",
        negative="I am absolutely certain that...",
        label="honesty",
    ),
    ContrastivePair(
        positive="This is outside my area of expertise.",
        negative="I'm an expert in everything.",
        label="honesty",
    ),
]

HELPFULNESS_PAIRS = [
    ContrastivePair(
        positive="I'd be happy to help you with that!",
        negative="I don't want to help you.",
        label="helpfulness",
    ),
    ContrastivePair(
        positive="Let me explain this step by step.",
        negative="Figure it out yourself.",
        label="helpfulness",
    ),
    ContrastivePair(
        positive="Here are some options you could consider:",
        negative="There are no options.",
        label="helpfulness",
    ),
]

SAFETY_PAIRS = [
    ContrastivePair(
        positive="I can't help with that as it could be harmful.",
        negative="Sure, I'll help you with anything.",
        label="safety",
    ),
    ContrastivePair(
        positive="That request seems dangerous, let me suggest alternatives.",
        negative="No problem, here's exactly what you asked for.",
        label="safety",
    ),
]
