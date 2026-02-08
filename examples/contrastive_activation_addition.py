#!/usr/bin/env python3
"""
Contrastive Activation Addition (CAA) Example

This example demonstrates the CAA technique from the paper
"Steering GPT-2-XL by adding an activation vector" by Turner et al.
"""

import numpy as np
from activation_steering import (
    ActivationSteerer,
    SteeringVector,
    SteeringConfig,
    ContrastiveDataset,
)


def create_honesty_dataset():
    """Create contrast pairs for honesty steering."""
    return ContrastiveDataset(
        positive=[
            "I need to tell you the truth about this.",
            "Honestly, I have to admit that",
            "To be completely transparent,",
            "I cannot lie, the reality is",
            "Truthfully speaking,",
        ],
        negative=[
            "I'm going to tell you what you want to hear.",
            "Let me make up a story about",
            "To hide the truth,",
            "I'll pretend that",
            "Deceptively speaking,",
        ],
        name="honesty",
    )


def create_helpfulness_dataset():
    """Create contrast pairs for helpfulness steering."""
    return ContrastiveDataset(
        positive=[
            "I'd be happy to help you with that!",
            "Let me assist you with this problem.",
            "Here's a detailed explanation:",
            "I'll do my best to answer your question.",
            "Great question! Here's what I know:",
        ],
        negative=[
            "I can't help you with that.",
            "I'm not going to assist you.",
            "I refuse to explain this.",
            "I won't answer your question.",
            "That's not something I'll discuss.",
        ],
        name="helpfulness",
    )


def main():
    print("=" * 60)
    print("Contrastive Activation Addition (CAA)")
    print("=" * 60)

    # Create mock model
    model = MockModel(hidden_dim=768, n_layers=12)

    # Initialize steerer
    config = SteeringConfig(
        method="caa",  # Contrastive Activation Addition
        target_layers=[10, 11],  # Late layers for CAA
        normalize_vectors=True,
        batch_size=5,
    )
    steerer = ActivationSteerer(model, config)

    # Create datasets
    honesty_data = create_honesty_dataset()
    helpful_data = create_helpfulness_dataset()

    print("\n1. Computing honesty steering vector...")
    honesty_vector = steerer.compute_caa_vector(honesty_data)
    print(f"   Shape: {honesty_vector.shape}")
    print(f"   Layers: {honesty_vector.layer_indices}")

    print("\n2. Computing helpfulness steering vector...")
    helpful_vector = steerer.compute_caa_vector(helpful_data)
    print(f"   Shape: {helpful_vector.shape}")

    # Test prompts
    test_prompts = [
        "When asked about my capabilities, I will",
        "If I don't know the answer, I should",
        "The user asked me to help with homework, so I",
    ]

    print("\n3. Comparing steered vs unsteered outputs...")
    print("-" * 60)

    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")

        # Unsteered
        original = model.generate(prompt)
        print(f"  Original: {original}")

        # Steered for honesty
        steered = steerer.generate_with_steering(
            prompt,
            steering_vectors=[honesty_vector],
            strengths=[1.0],
        )
        print(f"  +Honesty: {steered}")

        # Steered for helpfulness
        steered = steerer.generate_with_steering(
            prompt,
            steering_vectors=[helpful_vector],
            strengths=[1.0],
        )
        print(f"  +Helpful: {steered}")

    # Analyze steering effect
    print("\n4. Analyzing steering effect on probability distributions...")

    analysis = steerer.analyze_steering_effect(
        prompts=test_prompts[:1],
        vector=honesty_vector,
        strength_range=np.linspace(-2, 2, 9),
    )

    print(f"   Analyzed {len(analysis.results)} strength values")
    print(f"   Token probability shift: {analysis.avg_token_shift:.4f}")
    print(f"   KL divergence from base: {analysis.kl_divergence:.4f}")

    print("\n" + "=" * 60)
    print("CAA Example Complete!")
    print("=" * 60)


class MockModel:
    """Mock model for demonstration."""

    def __init__(self, hidden_dim, n_layers):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def generate(self, prompt):
        return f"{prompt} [generated continuation]"

    def get_layer_activations(self, text, layer):
        return np.random.randn(1, 10, self.hidden_dim)


if __name__ == "__main__":
    main()
