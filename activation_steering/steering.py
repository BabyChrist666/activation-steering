"""
High-level steering interface.

Provides a simple API for applying activation steering
to language models during inference.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple
from contextlib import contextmanager

from activation_steering.vectors import SteeringVector
from activation_steering.hooks import (
    SteeringHook,
    CachingHook,
    AblationHook,
    attach_hooks,
    remove_hooks,
)


@dataclass
class SteeringConfig:
    """Configuration for activation steering."""
    coefficient: float = 1.0
    position: str = "all"  # "all", "last", "first"
    normalize: bool = True
    layers: Optional[List[int]] = None  # None = use vector's layer
    combine_method: str = "add"  # "add", "interpolate"


@dataclass
class SteeringResult:
    """Result of a steered generation."""
    text: str
    original_text: Optional[str] = None
    steering_applied: bool = False
    config: Optional[SteeringConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "original_text": self.original_text,
            "steering_applied": self.steering_applied,
            "config": {
                "coefficient": self.config.coefficient,
                "position": self.config.position,
            } if self.config else None,
            "metadata": self.metadata,
        }


class ActivationSteerer:
    """
    High-level interface for activation steering.

    Provides methods for applying steering vectors to models
    during text generation or inference.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        default_config: Optional[SteeringConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.default_config = default_config or SteeringConfig()
        self._active_hooks: List[SteeringHook] = []
        self._device = next(model.parameters()).device

    @property
    def device(self) -> torch.device:
        return self._device

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

    @contextmanager
    def steering_context(
        self,
        vectors: Union[SteeringVector, List[SteeringVector]],
        config: Optional[SteeringConfig] = None,
    ):
        """
        Context manager for applying steering.

        Example:
            with steerer.steering_context(honesty_vector, config):
                output = model.generate(...)
        """
        config = config or self.default_config

        if isinstance(vectors, SteeringVector):
            vectors = [vectors]

        hooks = []
        blocks = self._get_blocks()

        try:
            for vec in vectors:
                layers = config.layers if config.layers else [vec.layer_idx]

                for layer_idx in layers:
                    if layer_idx >= len(blocks):
                        continue

                    hook = SteeringHook(
                        layer_idx=layer_idx,
                        steering_vector=vec.vector,
                        coefficient=config.coefficient,
                        position=config.position,
                        normalize=config.normalize,
                    )
                    hook.attach(blocks[layer_idx])
                    hooks.append(hook)

            self._active_hooks = hooks
            yield

        finally:
            for hook in hooks:
                hook.remove()
            self._active_hooks = []

    def generate(
        self,
        prompt: str,
        vectors: Optional[Union[SteeringVector, List[SteeringVector]]] = None,
        config: Optional[SteeringConfig] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        compare_unsteered: bool = False,
        **generate_kwargs,
    ) -> SteeringResult:
        """
        Generate text with optional steering.

        Args:
            prompt: Input prompt.
            vectors: Steering vector(s) to apply.
            config: Steering configuration.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample.
            compare_unsteered: Also generate without steering.
            **generate_kwargs: Additional args for model.generate().

        Returns:
            SteeringResult with generated text.
        """
        self.model.eval()
        config = config or self.default_config

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Generate without steering for comparison
        original_text = None
        if compare_unsteered:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **generate_kwargs,
                )
            original_text = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True,
            )

        # Generate with steering
        if vectors is not None:
            with self.steering_context(vectors, config):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.eos_token_id,
                        **generate_kwargs,
                    )
        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **generate_kwargs,
                )

        steered_text = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )

        return SteeringResult(
            text=steered_text,
            original_text=original_text,
            steering_applied=vectors is not None,
            config=config if vectors is not None else None,
            metadata={
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )

    def sweep_coefficients(
        self,
        prompt: str,
        vector: SteeringVector,
        coefficients: List[float],
        config: Optional[SteeringConfig] = None,
        **generate_kwargs,
    ) -> List[SteeringResult]:
        """
        Generate with multiple steering coefficients.

        Useful for finding the optimal steering strength.
        """
        results = []
        config = config or SteeringConfig()

        for coeff in coefficients:
            sweep_config = SteeringConfig(
                coefficient=coeff,
                position=config.position,
                normalize=config.normalize,
                layers=config.layers,
            )

            result = self.generate(
                prompt=prompt,
                vectors=vector,
                config=sweep_config,
                **generate_kwargs,
            )
            result.metadata["coefficient"] = coeff
            results.append(result)

        return results

    def get_activations(
        self,
        text: str,
        layer_idx: int,
        vectors: Optional[Union[SteeringVector, List[SteeringVector]]] = None,
        config: Optional[SteeringConfig] = None,
    ) -> torch.Tensor:
        """
        Get activations at a specific layer.

        Args:
            text: Input text.
            layer_idx: Layer to get activations from.
            vectors: Optional steering to apply.
            config: Steering configuration.

        Returns:
            Activation tensor of shape [batch, seq, hidden_dim].
        """
        self.model.eval()
        blocks = self._get_blocks()

        cache_hook = CachingHook(layer_idx)
        cache_hook.attach(blocks[layer_idx])

        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if vectors is not None:
                with self.steering_context(vectors, config):
                    with torch.no_grad():
                        self.model(**inputs)
            else:
                with torch.no_grad():
                    self.model(**inputs)

            return cache_hook.get_cached()

        finally:
            cache_hook.remove()

    def project_onto_vector(
        self,
        text: str,
        vector: SteeringVector,
    ) -> torch.Tensor:
        """
        Project activations onto a steering vector direction.

        Returns scalar projections for each position.
        """
        activations = self.get_activations(text, vector.layer_idx)
        return vector.project_onto(activations)


def create_steerer(
    model_name_or_path: str,
    device: Optional[str] = None,
    **model_kwargs,
) -> ActivationSteerer:
    """
    Create an ActivationSteerer from a model name.

    Args:
        model_name_or_path: HuggingFace model name or path.
        device: Device to load model on.
        **model_kwargs: Additional args for model loading.

    Returns:
        Configured ActivationSteerer.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("transformers library required for create_steerer")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )

    if device:
        model = model.to(device)

    return ActivationSteerer(model, tokenizer)
