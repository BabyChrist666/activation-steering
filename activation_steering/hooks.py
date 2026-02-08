"""
Activation hooks for steering and analysis.

Provides hook classes that can be attached to model layers
to modify or record activations during forward passes.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class HookResult:
    """Result from a hook operation."""
    layer_idx: int
    activation_shape: tuple
    modified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ActivationHook(ABC):
    """
    Base class for activation hooks.

    Hooks can be attached to model layers to intercept,
    modify, or record activations during forward passes.
    """

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._active = True
        self._call_count = 0

    @property
    def active(self) -> bool:
        return self._active

    def enable(self):
        self._active = True

    def disable(self):
        self._active = False

    @abstractmethod
    def hook_fn(
        self,
        module: nn.Module,
        inputs: tuple,
        outputs: Union[torch.Tensor, tuple],
    ) -> Optional[Union[torch.Tensor, tuple]]:
        """The hook function to be called during forward pass."""
        pass

    def attach(self, module: nn.Module) -> "ActivationHook":
        """Attach the hook to a module."""
        self._handle = module.register_forward_hook(self._wrapped_hook_fn)
        return self

    def _wrapped_hook_fn(
        self,
        module: nn.Module,
        inputs: tuple,
        outputs: Union[torch.Tensor, tuple],
    ) -> Optional[Union[torch.Tensor, tuple]]:
        """Wrapper that checks if hook is active."""
        if not self._active:
            return None
        self._call_count += 1
        return self.hook_fn(module, inputs, outputs)

    def remove(self):
        """Remove the hook from the module."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


class SteeringHook(ActivationHook):
    """
    Hook that adds a steering vector to activations.

    The steering vector is added to the residual stream
    to steer the model's behavior in a particular direction.
    """

    def __init__(
        self,
        layer_idx: int,
        steering_vector: torch.Tensor,
        coefficient: float = 1.0,
        position: str = "all",  # "all", "last", "first"
        normalize: bool = False,
    ):
        super().__init__(layer_idx)
        self.steering_vector = steering_vector
        self.coefficient = coefficient
        self.position = position
        self.normalize = normalize

        if normalize:
            self.steering_vector = self.steering_vector / self.steering_vector.norm()

    def set_coefficient(self, coeff: float):
        """Update the steering coefficient."""
        self.coefficient = coeff

    def hook_fn(
        self,
        module: nn.Module,
        inputs: tuple,
        outputs: Union[torch.Tensor, tuple],
    ) -> Union[torch.Tensor, tuple]:
        # Extract the tensor from outputs
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            rest = outputs[1:]
        else:
            hidden_states = outputs
            rest = None

        # Ensure steering vector is on same device
        steering = self.steering_vector.to(hidden_states.device)
        steering = steering * self.coefficient

        # Apply steering based on position
        if self.position == "all":
            hidden_states = hidden_states + steering
        elif self.position == "last":
            hidden_states[:, -1, :] = hidden_states[:, -1, :] + steering
        elif self.position == "first":
            hidden_states[:, 0, :] = hidden_states[:, 0, :] + steering
        else:
            raise ValueError(f"Unknown position: {self.position}")

        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states


class CachingHook(ActivationHook):
    """
    Hook that caches activations for later analysis.

    Useful for collecting activations across multiple
    forward passes for analysis or vector extraction.
    """

    def __init__(
        self,
        layer_idx: int,
        max_cache_size: int = 1000,
        store_inputs: bool = False,
    ):
        super().__init__(layer_idx)
        self.max_cache_size = max_cache_size
        self.store_inputs = store_inputs
        self.cache: List[torch.Tensor] = []
        self.input_cache: List[torch.Tensor] = []

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.input_cache.clear()

    def get_cached(self) -> torch.Tensor:
        """Get all cached activations as a single tensor."""
        if not self.cache:
            raise ValueError("Cache is empty")
        return torch.cat(self.cache, dim=0)

    def hook_fn(
        self,
        module: nn.Module,
        inputs: tuple,
        outputs: Union[torch.Tensor, tuple],
    ) -> None:
        # Extract the tensor from outputs
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        # Store activation (detached to avoid memory issues)
        if len(self.cache) < self.max_cache_size:
            self.cache.append(hidden_states.detach().cpu())

        if self.store_inputs and len(self.input_cache) < self.max_cache_size:
            if len(inputs) > 0:
                inp = inputs[0] if isinstance(inputs[0], torch.Tensor) else inputs
                if isinstance(inp, torch.Tensor):
                    self.input_cache.append(inp.detach().cpu())

        return None  # Don't modify outputs


class AblationHook(ActivationHook):
    """
    Hook that ablates (zeros out) specific directions.

    Useful for understanding the causal role of
    particular features or directions in the model.
    """

    def __init__(
        self,
        layer_idx: int,
        directions: Optional[torch.Tensor] = None,
        neuron_indices: Optional[List[int]] = None,
        ablation_type: str = "zero",  # "zero", "mean", "resample"
    ):
        super().__init__(layer_idx)
        self.directions = directions
        self.neuron_indices = neuron_indices
        self.ablation_type = ablation_type
        self._mean_cache: Optional[torch.Tensor] = None

    def set_mean(self, mean: torch.Tensor):
        """Set the mean for mean ablation."""
        self._mean_cache = mean

    def hook_fn(
        self,
        module: nn.Module,
        inputs: tuple,
        outputs: Union[torch.Tensor, tuple],
    ) -> Union[torch.Tensor, tuple]:
        # Extract the tensor from outputs
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            rest = outputs[1:]
        else:
            hidden_states = outputs
            rest = None

        # Ablate specific neurons
        if self.neuron_indices is not None:
            if self.ablation_type == "zero":
                hidden_states = hidden_states.clone()
                hidden_states[..., self.neuron_indices] = 0
            elif self.ablation_type == "mean" and self._mean_cache is not None:
                hidden_states = hidden_states.clone()
                hidden_states[..., self.neuron_indices] = self._mean_cache[self.neuron_indices]

        # Ablate specific directions
        if self.directions is not None:
            directions = self.directions.to(hidden_states.device)
            # Project out the directions
            for direction in directions:
                direction = direction / direction.norm()
                proj = torch.einsum("...d,d->...", hidden_states, direction)
                hidden_states = hidden_states - proj.unsqueeze(-1) * direction

        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states


class CompositeHook(ActivationHook):
    """
    Hook that combines multiple hooks.

    Applies hooks in sequence, allowing complex
    activation modifications.
    """

    def __init__(self, layer_idx: int, hooks: List[ActivationHook]):
        super().__init__(layer_idx)
        self.hooks = hooks

    def hook_fn(
        self,
        module: nn.Module,
        inputs: tuple,
        outputs: Union[torch.Tensor, tuple],
    ) -> Optional[Union[torch.Tensor, tuple]]:
        current_outputs = outputs

        for hook in self.hooks:
            if hook.active:
                result = hook.hook_fn(module, inputs, current_outputs)
                if result is not None:
                    current_outputs = result

        return current_outputs


class ConditionalHook(ActivationHook):
    """
    Hook that only activates based on a condition.

    Useful for applying steering only when certain
    patterns are detected in the input.
    """

    def __init__(
        self,
        layer_idx: int,
        inner_hook: ActivationHook,
        condition_fn: Callable[[torch.Tensor], bool],
    ):
        super().__init__(layer_idx)
        self.inner_hook = inner_hook
        self.condition_fn = condition_fn

    def hook_fn(
        self,
        module: nn.Module,
        inputs: tuple,
        outputs: Union[torch.Tensor, tuple],
    ) -> Optional[Union[torch.Tensor, tuple]]:
        # Extract the tensor from outputs
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        if self.condition_fn(hidden_states):
            return self.inner_hook.hook_fn(module, inputs, outputs)

        return None


def attach_hooks(
    model: nn.Module,
    hooks: List[ActivationHook],
) -> List[ActivationHook]:
    """
    Attach multiple hooks to a model.

    Args:
        model: The model to attach hooks to.
        hooks: List of hooks to attach.

    Returns:
        List of attached hooks.
    """
    # Find layer blocks
    if hasattr(model, "blocks"):
        blocks = model.blocks
    elif hasattr(model, "transformer"):
        if hasattr(model.transformer, "h"):
            blocks = model.transformer.h
        else:
            blocks = model.transformer.blocks
    elif hasattr(model, "layers"):
        blocks = model.layers
    else:
        raise ValueError("Could not find layer blocks in model")

    for hook in hooks:
        if hook.layer_idx >= len(blocks):
            raise ValueError(f"Layer {hook.layer_idx} does not exist (max: {len(blocks)-1})")
        hook.attach(blocks[hook.layer_idx])

    return hooks


def remove_hooks(hooks: List[ActivationHook]) -> None:
    """Remove all hooks from the list."""
    for hook in hooks:
        hook.remove()
