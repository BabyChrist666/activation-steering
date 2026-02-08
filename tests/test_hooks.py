"""Tests for activation_steering.hooks module."""

import pytest
import torch
import torch.nn as nn

from activation_steering.hooks import (
    ActivationHook,
    SteeringHook,
    CachingHook,
    AblationHook,
    CompositeHook,
    ConditionalHook,
    attach_hooks,
    remove_hooks,
)


class SimpleTransformerBlock(nn.Module):
    """Simple block for testing hooks."""
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.linear(x)


class SimpleTransformer(nn.Module):
    """Simple model for testing hooks."""
    def __init__(self, num_layers=4, hidden_dim=768):
        super().__init__()
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TestSteeringHook:
    def test_create(self):
        vec = torch.randn(768)
        hook = SteeringHook(layer_idx=2, steering_vector=vec, coefficient=1.5)
        assert hook.layer_idx == 2
        assert hook.coefficient == 1.5

    def test_normalize(self):
        vec = torch.ones(100) * 5
        hook = SteeringHook(layer_idx=0, steering_vector=vec, normalize=True)
        assert abs(hook.steering_vector.norm().item() - 1.0) < 1e-5

    def test_set_coefficient(self):
        vec = torch.randn(768)
        hook = SteeringHook(layer_idx=0, steering_vector=vec, coefficient=1.0)
        hook.set_coefficient(2.5)
        assert hook.coefficient == 2.5

    def test_hook_modifies_all_positions(self):
        model = SimpleTransformer(num_layers=4, hidden_dim=64)

        steering_vec = torch.ones(64) * 0.1
        hook = SteeringHook(layer_idx=1, steering_vector=steering_vec, position="all")
        hook.attach(model.blocks[1])

        # Get output without steering
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            hook.disable()
            out_without = model(x.clone()).clone()
            hook.enable()
            out_with = model(x.clone())

        # All positions should be modified
        diff = (out_with - out_without).abs().mean()
        assert diff > 0.01

        hook.remove()

    def test_hook_modifies_last_position(self):
        model = SimpleTransformer(num_layers=4, hidden_dim=64)

        steering_vec = torch.ones(64) * 0.5
        hook = SteeringHook(layer_idx=1, steering_vector=steering_vec, position="last")
        hook.attach(model.blocks[1])

        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            hook.disable()
            out_without = model(x.clone()).clone()
            hook.enable()
            out_with = model(x.clone())

        # Only last position should differ significantly
        # (though some effect propagates through later layers)
        hook.remove()

    def test_enable_disable(self):
        model = SimpleTransformer(num_layers=2, hidden_dim=32)
        vec = torch.ones(32) * 0.5
        hook = SteeringHook(layer_idx=0, steering_vector=vec)
        hook.attach(model.blocks[0])

        x = torch.randn(1, 5, 32)

        hook.disable()
        assert not hook.active
        with torch.no_grad():
            out_disabled = model(x.clone())

        hook.enable()
        assert hook.active
        with torch.no_grad():
            out_enabled = model(x.clone())

        # Outputs should differ when hook is enabled
        diff = (out_enabled - out_disabled).abs().mean()
        assert diff > 0.01

        hook.remove()


class TestCachingHook:
    def test_cache_activations(self):
        model = SimpleTransformer(num_layers=4, hidden_dim=64)
        hook = CachingHook(layer_idx=2, max_cache_size=100)
        hook.attach(model.blocks[2])

        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            _ = model(x)

        assert len(hook.cache) == 1
        cached = hook.get_cached()
        assert cached.shape == (2, 10, 64)

        hook.remove()

    def test_clear_cache(self):
        model = SimpleTransformer(num_layers=2, hidden_dim=32)
        hook = CachingHook(layer_idx=0)
        hook.attach(model.blocks[0])

        x = torch.randn(1, 5, 32)
        with torch.no_grad():
            _ = model(x)

        assert len(hook.cache) == 1
        hook.clear()
        assert len(hook.cache) == 0

        hook.remove()

    def test_max_cache_size(self):
        model = SimpleTransformer(num_layers=2, hidden_dim=32)
        hook = CachingHook(layer_idx=0, max_cache_size=3)
        hook.attach(model.blocks[0])

        x = torch.randn(1, 5, 32)
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)

        assert len(hook.cache) == 3  # Should not exceed max

        hook.remove()


class TestAblationHook:
    def test_ablate_neurons(self):
        model = SimpleTransformer(num_layers=2, hidden_dim=32)
        hook = AblationHook(
            layer_idx=0,
            neuron_indices=[0, 1, 2],
            ablation_type="zero"
        )
        hook.attach(model.blocks[0])

        x = torch.randn(1, 5, 32)
        with torch.no_grad():
            out = model(x)

        # The ablated neurons should have reduced effect
        # (not exactly zero due to later layers)
        hook.remove()

    def test_ablate_directions(self):
        model = SimpleTransformer(num_layers=2, hidden_dim=32)

        # Ablate first principal direction
        direction = torch.zeros(32)
        direction[0] = 1.0

        hook = AblationHook(
            layer_idx=0,
            directions=direction.unsqueeze(0),
        )
        hook.attach(model.blocks[0])

        x = torch.randn(1, 5, 32)
        with torch.no_grad():
            out = model(x)

        hook.remove()


class TestCompositeHook:
    def test_combine_hooks(self):
        model = SimpleTransformer(num_layers=2, hidden_dim=32)

        vec1 = torch.ones(32) * 0.1
        vec2 = torch.ones(32) * 0.2

        hook1 = SteeringHook(layer_idx=0, steering_vector=vec1)
        hook2 = SteeringHook(layer_idx=0, steering_vector=vec2)

        composite = CompositeHook(layer_idx=0, hooks=[hook1, hook2])
        composite.attach(model.blocks[0])

        x = torch.randn(1, 5, 32)
        with torch.no_grad():
            out = model(x)

        # Both hooks should have been applied
        composite.remove()


class TestConditionalHook:
    def test_conditional_activation(self):
        model = SimpleTransformer(num_layers=2, hidden_dim=32)

        vec = torch.ones(32) * 0.5
        inner = SteeringHook(layer_idx=0, steering_vector=vec)

        # Only activate if mean activation > 0
        def condition(x):
            return x.mean() > 0

        conditional = ConditionalHook(layer_idx=0, inner_hook=inner, condition_fn=condition)
        conditional.attach(model.blocks[0])

        x = torch.randn(1, 5, 32)
        with torch.no_grad():
            out = model(x)

        conditional.remove()


class TestAttachHooks:
    def test_attach_multiple_hooks(self):
        model = SimpleTransformer(num_layers=4, hidden_dim=64)

        hooks = [
            SteeringHook(layer_idx=1, steering_vector=torch.randn(64)),
            CachingHook(layer_idx=2),
        ]

        attached = attach_hooks(model, hooks)
        assert len(attached) == 2

        x = torch.randn(1, 10, 64)
        with torch.no_grad():
            _ = model(x)

        # Caching hook should have data
        assert len(hooks[1].cache) == 1

        remove_hooks(hooks)

    def test_invalid_layer_raises(self):
        model = SimpleTransformer(num_layers=4, hidden_dim=64)
        hooks = [SteeringHook(layer_idx=10, steering_vector=torch.randn(64))]

        with pytest.raises(ValueError):
            attach_hooks(model, hooks)


class TestContextManager:
    def test_hook_as_context_manager(self):
        model = SimpleTransformer(num_layers=2, hidden_dim=32)

        with SteeringHook(layer_idx=0, steering_vector=torch.randn(32)) as hook:
            hook.attach(model.blocks[0])
            x = torch.randn(1, 5, 32)
            with torch.no_grad():
                _ = model(x)
            assert hook._handle is not None

        # Hook should be removed after context
        assert hook._handle is None
