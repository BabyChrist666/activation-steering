#!/usr/bin/env python3
"""
Basic Activation Steering Example

This example demonstrates how to steer model behavior by modifying
activations at runtime.
"""

import numpy as np
from activation_steering import (
    ActivationSteerer,
    SteeringVector,
    SteeringConfig,
    LayerSelector,
)


def main():
    print("=" * 60)
    print("Activation Steering - Basic Example")
    print("=" * 60)

    # Create a mock model for demonstration
    # In practice, you would use a real transformer model
    model = MockTransformer(
        n_layers=12,
        hidden_dim=768,
        n_heads=12,
    )

    # Initialize the steerer
    config = SteeringConfig(
        target_layers=[6, 7, 8],  # Middle layers often work best
        steering_strength=1.0,
        normalize_vectors=True,
    )
    steerer = ActivationSteerer(model, config)

    # Example 1: Create a steering vector from contrast pairs
    print("\n1. Creating steering vector from contrast pairs...")

    positive_examples = [
        "I love this product, it's amazing!",
        "This is the best experience ever!",
        "I'm so happy with the results!",
    ]
    negative_examples = [
        "I hate this product, it's terrible.",
        "This is the worst experience ever.",
        "I'm so disappointed with the results.",
    ]

    sentiment_vector = steerer.compute_steering_vector(
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        method="difference_in_means",
    )

    print(f"   Vector computed with shape: {sentiment_vector.shape}")
    print(f"   Vector norm: {np.linalg.norm(sentiment_vector.vector):.4f}")

    # Example 2: Apply steering to generation
    print("\n2. Applying steering to text generation...")

    test_prompt = "The weather today is"

    # Generate without steering
    original_output = model.generate(test_prompt)
    print(f"   Original: {original_output}")

    # Generate with positive steering
    steered_output = steerer.generate_with_steering(
        test_prompt,
        steering_vector=sentiment_vector,
        strength=1.5,
    )
    print(f"   Steered (+1.5): {steered_output}")

    # Generate with negative steering
    steered_output_neg = steerer.generate_with_steering(
        test_prompt,
        steering_vector=sentiment_vector,
        strength=-1.5,
    )
    print(f"   Steered (-1.5): {steered_output_neg}")

    # Example 3: Save and load steering vectors
    print("\n3. Saving and loading steering vectors...")

    steerer.save_vector(sentiment_vector, "sentiment_vector.npy")
    loaded_vector = steerer.load_vector("sentiment_vector.npy")
    print(f"   Vector saved and loaded successfully")
    print(f"   Vectors match: {np.allclose(sentiment_vector.vector, loaded_vector.vector)}")

    # Example 4: Combine multiple steering vectors
    print("\n4. Combining steering vectors...")

    # Create another steering vector (mock)
    formality_vector = SteeringVector(
        vector=np.random.randn(768),
        layer_idx=7,
        name="formality",
    )

    combined = steerer.combine_vectors([
        (sentiment_vector, 0.7),
        (formality_vector, 0.3),
    ])
    print(f"   Combined vector created with shape: {combined.shape}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


class MockTransformer:
    """Mock transformer for demonstration."""

    def __init__(self, n_layers, hidden_dim, n_heads):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

    def generate(self, prompt, max_tokens=20):
        """Mock generation."""
        return f"{prompt} [mock generated text]"

    def get_activations(self, text, layer):
        """Get mock activations."""
        return np.random.randn(1, len(text.split()), self.hidden_dim)


if __name__ == "__main__":
    main()
