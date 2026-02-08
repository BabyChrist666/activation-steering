"""
Analysis tools for activation steering.

Provides methods for understanding steering vectors,
measuring their effects, and visualizing activations.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from activation_steering.vectors import SteeringVector, ContrastivePair
from activation_steering.hooks import CachingHook, attach_hooks, remove_hooks


@dataclass
class ProjectionAnalysis:
    """Analysis of projections onto a steering direction."""
    projections: torch.Tensor
    mean_projection: float
    std_projection: float
    min_projection: float
    max_projection: float
    layer_idx: int

    def to_dict(self) -> dict:
        return {
            "mean_projection": self.mean_projection,
            "std_projection": self.std_projection,
            "min_projection": self.min_projection,
            "max_projection": self.max_projection,
            "layer_idx": self.layer_idx,
        }


@dataclass
class SteeringEffectiveness:
    """Measures of how effective a steering vector is."""
    projection_shift: float  # Change in projection before/after steering
    cosine_similarity: float  # Similarity to steering direction
    magnitude_ratio: float  # Ratio of steering to original activation magnitude
    layer_idx: int
    coefficient: float

    def to_dict(self) -> dict:
        return {
            "projection_shift": self.projection_shift,
            "cosine_similarity": self.cosine_similarity,
            "magnitude_ratio": self.magnitude_ratio,
            "layer_idx": self.layer_idx,
            "coefficient": self.coefficient,
        }


class ActivationAnalyzer:
    """
    Tools for analyzing activations and steering effects.
    """

    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._device = next(model.parameters()).device

    def _get_blocks(self):
        """Get the layer blocks from the model."""
        if hasattr(self.model, "blocks"):
            return self.model.blocks
        elif hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "h"):
                return self.model.transformer.h
            else:
                return self.model.transformer.blocks
        elif hasattr(self.model, "layers"):
            return self.model.layers
        else:
            raise ValueError("Could not find layer blocks in model")

    def analyze_projection(
        self,
        texts: List[str],
        vector: SteeringVector,
    ) -> ProjectionAnalysis:
        """
        Analyze how texts project onto a steering direction.

        Args:
            texts: List of texts to analyze.
            vector: Steering vector to project onto.

        Returns:
            ProjectionAnalysis with statistics.
        """
        self.model.eval()
        blocks = self._get_blocks()

        all_projections = []

        for text in texts:
            cache_hook = CachingHook(vector.layer_idx)
            cache_hook.attach(blocks[vector.layer_idx])

            try:
                inputs = self.tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with torch.no_grad():
                    self.model(**inputs)

                activations = cache_hook.get_cached()
                projections = vector.project_onto(activations)
                all_projections.append(projections.flatten())

            finally:
                cache_hook.remove()

        all_projections = torch.cat(all_projections)

        return ProjectionAnalysis(
            projections=all_projections,
            mean_projection=all_projections.mean().item(),
            std_projection=all_projections.std().item(),
            min_projection=all_projections.min().item(),
            max_projection=all_projections.max().item(),
            layer_idx=vector.layer_idx,
        )

    def compare_projections(
        self,
        positive_texts: List[str],
        negative_texts: List[str],
        vector: SteeringVector,
    ) -> Dict[str, Any]:
        """
        Compare projections between positive and negative examples.

        Args:
            positive_texts: Texts expected to project positively.
            negative_texts: Texts expected to project negatively.
            vector: Steering vector.

        Returns:
            Comparison statistics.
        """
        pos_analysis = self.analyze_projection(positive_texts, vector)
        neg_analysis = self.analyze_projection(negative_texts, vector)

        separation = pos_analysis.mean_projection - neg_analysis.mean_projection
        pooled_std = np.sqrt(
            (pos_analysis.std_projection**2 + neg_analysis.std_projection**2) / 2
        )

        return {
            "positive_mean": pos_analysis.mean_projection,
            "negative_mean": neg_analysis.mean_projection,
            "separation": separation,
            "effect_size": separation / pooled_std if pooled_std > 0 else 0,
            "positive_std": pos_analysis.std_projection,
            "negative_std": neg_analysis.std_projection,
        }

    def find_best_layer(
        self,
        pairs: List[ContrastivePair],
        layer_range: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, float]:
        """
        Find the layer with best separation between contrastive pairs.

        Args:
            pairs: Contrastive pairs to test.
            layer_range: (start, end) layers to test.

        Returns:
            (best_layer_idx, separation_score)
        """
        blocks = self._get_blocks()
        num_layers = len(blocks)

        if layer_range is None:
            layer_range = (0, num_layers)

        start, end = layer_range
        end = min(end, num_layers)

        best_layer = start
        best_separation = 0

        for layer_idx in range(start, end):
            # Create a simple steering vector for this layer
            from activation_steering.vectors import extract_steering_vector

            try:
                vector = extract_steering_vector(
                    self.model,
                    self.tokenizer,
                    pairs[:min(3, len(pairs))],  # Use subset for speed
                    layer_idx,
                )

                positive_texts = [p.positive for p in pairs]
                negative_texts = [p.negative for p in pairs]

                comparison = self.compare_projections(
                    positive_texts, negative_texts, vector
                )

                if abs(comparison["separation"]) > abs(best_separation):
                    best_separation = comparison["separation"]
                    best_layer = layer_idx

            except Exception:
                continue

        return best_layer, best_separation

    def measure_steering_effect(
        self,
        text: str,
        vector: SteeringVector,
        coefficient: float = 1.0,
    ) -> SteeringEffectiveness:
        """
        Measure the effect of applying a steering vector.

        Args:
            text: Text to analyze.
            vector: Steering vector to apply.
            coefficient: Steering coefficient.

        Returns:
            SteeringEffectiveness metrics.
        """
        self.model.eval()
        blocks = self._get_blocks()

        # Get activations without steering
        cache_hook_before = CachingHook(vector.layer_idx)
        cache_hook_before.attach(blocks[vector.layer_idx])

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            self.model(**inputs)

        before_acts = cache_hook_before.get_cached()
        cache_hook_before.remove()

        # Get activations with steering
        from activation_steering.hooks import SteeringHook

        steering_hook = SteeringHook(
            vector.layer_idx,
            vector.vector,
            coefficient=coefficient,
        )
        steering_hook.attach(blocks[vector.layer_idx])

        cache_hook_after = CachingHook(vector.layer_idx)
        # Attach after steering hook
        cache_hook_after.attach(blocks[vector.layer_idx])

        with torch.no_grad():
            self.model(**inputs)

        after_acts = cache_hook_after.get_cached()

        steering_hook.remove()
        cache_hook_after.remove()

        # Compute metrics
        before_proj = vector.project_onto(before_acts).mean().item()
        after_proj = vector.project_onto(after_acts).mean().item()

        vec_norm = vector.vector / vector.vector.norm()
        before_mean = before_acts.mean(dim=(0, 1))
        cosine_sim = torch.nn.functional.cosine_similarity(
            before_mean.unsqueeze(0).to(vec_norm.device),
            vec_norm.unsqueeze(0),
        ).item()

        before_mag = before_acts.norm(dim=-1).mean().item()
        steering_mag = (vector.vector * coefficient).norm().item()

        return SteeringEffectiveness(
            projection_shift=after_proj - before_proj,
            cosine_similarity=cosine_sim,
            magnitude_ratio=steering_mag / before_mag if before_mag > 0 else 0,
            layer_idx=vector.layer_idx,
            coefficient=coefficient,
        )

    def activation_statistics(
        self,
        texts: List[str],
        layer_idx: int,
    ) -> Dict[str, Any]:
        """
        Compute activation statistics at a layer.

        Args:
            texts: Texts to analyze.
            layer_idx: Layer to analyze.

        Returns:
            Dictionary of statistics.
        """
        self.model.eval()
        blocks = self._get_blocks()

        all_acts = []

        for text in texts:
            cache_hook = CachingHook(layer_idx)
            cache_hook.attach(blocks[layer_idx])

            try:
                inputs = self.tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with torch.no_grad():
                    self.model(**inputs)

                acts = cache_hook.get_cached()
                all_acts.append(acts.flatten(0, 1))  # Merge batch and seq

            finally:
                cache_hook.remove()

        all_acts = torch.cat(all_acts, dim=0)

        return {
            "mean": all_acts.mean(dim=0).tolist(),
            "std": all_acts.std(dim=0).tolist(),
            "global_mean": all_acts.mean().item(),
            "global_std": all_acts.std().item(),
            "min": all_acts.min().item(),
            "max": all_acts.max().item(),
            "shape": list(all_acts.shape),
            "layer_idx": layer_idx,
        }

    def vector_similarity(
        self,
        vec1: SteeringVector,
        vec2: SteeringVector,
    ) -> float:
        """Compute cosine similarity between two steering vectors."""
        v1 = vec1.vector.flatten()
        v2 = vec2.vector.flatten()

        if v1.device != v2.device:
            v2 = v2.to(v1.device)

        return torch.nn.functional.cosine_similarity(
            v1.unsqueeze(0),
            v2.unsqueeze(0),
        ).item()

    def decompose_vector(
        self,
        vector: SteeringVector,
        basis_vectors: List[SteeringVector],
    ) -> Dict[str, float]:
        """
        Decompose a steering vector onto a basis of other vectors.

        Returns coefficients for each basis vector.
        """
        coeffs = {}
        remaining = vector.vector.clone()

        for basis in basis_vectors:
            basis_norm = basis.vector / basis.vector.norm()
            if basis_norm.device != remaining.device:
                basis_norm = basis_norm.to(remaining.device)

            coeff = torch.dot(remaining.flatten(), basis_norm.flatten()).item()
            coeffs[basis.name or f"basis_{len(coeffs)}"] = coeff

            # Project out this component
            remaining = remaining - coeff * basis_norm

        coeffs["residual_norm"] = remaining.norm().item()
        coeffs["original_norm"] = vector.norm

        return coeffs
