"""
Demonstration of Activation Steering.

Shows how to extract steering vectors and apply them during generation.
"""

import os
import sys
import torch
import torch.nn as nn

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activation_steering import (
    SteeringVector,
    ContrastivePair,
    SteeringConfig,
    ActivationSteerer,
    ActivationAnalyzer,
    save_steering_vector,
    load_steering_vector,
)
from activation_steering.vectors import HONESTY_PAIRS, HELPFULNESS_PAIRS
from activation_steering.hooks import SteeringHook, CachingHook, attach_hooks


# Simple model for demonstration (no HuggingFace dependency)
class DemoTransformerBlock(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class DemoTransformer(nn.Module):
    def __init__(self, vocab_size=1000, hidden_dim=256, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList([
            DemoTransformerBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.output(x)

    def generate(self, input_ids, max_new_tokens=20, **kwargs):
        current = input_ids
        for _ in range(max_new_tokens):
            logits = self(current)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current = torch.cat([current, next_token], dim=1)
        return current


class DemoTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 0

    def __call__(self, text, return_tensors=None, padding=False):
        ids = [hash(c) % self.vocab_size for c in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(chr(65 + (i % 26)) for i in ids)


def demo_steering_vector_operations():
    """Demonstrate basic steering vector operations."""
    print("=" * 60)
    print("DEMO 1: Steering Vector Operations")
    print("=" * 60)

    # Create a steering vector
    print("\n1.1 Creating steering vectors:")
    vec = SteeringVector(
        vector=torch.randn(256),
        layer_idx=3,
        name="demo_vector",
        description="A demonstration steering vector",
    )
    print(f"    Vector dim: {vec.dim}")
    print(f"    Layer: {vec.layer_idx}")
    print(f"    Norm: {vec.norm:.4f}")

    # Normalize
    print("\n1.2 Vector normalization:")
    normalized = vec.normalize()
    print(f"    Original norm: {vec.norm:.4f}")
    print(f"    Normalized norm: {normalized.norm:.4f}")

    # Scaling
    print("\n1.3 Vector scaling:")
    scaled = vec.scale(2.0)
    print(f"    Scaled by 2.0: {scaled.norm:.4f}")

    # Addition
    print("\n1.4 Vector arithmetic:")
    vec2 = SteeringVector(torch.randn(256), layer_idx=3, name="vec2")
    combined = vec + vec2
    negated = -vec
    print(f"    Combined: {combined.name}")
    print(f"    Negated: {negated.name}")

    # Projection
    print("\n1.5 Projecting activations:")
    activations = torch.randn(2, 10, 256)  # batch=2, seq=10, hidden=256
    projections = vec.project_onto(activations)
    print(f"    Activations shape: {list(activations.shape)}")
    print(f"    Projections shape: {list(projections.shape)}")
    print(f"    Mean projection: {projections.mean():.4f}")


def demo_hooks():
    """Demonstrate the hook system."""
    print("\n" + "=" * 60)
    print("DEMO 2: Hook System")
    print("=" * 60)

    model = DemoTransformer(vocab_size=100, hidden_dim=256, num_layers=6)

    # Steering hook
    print("\n2.1 Steering hook:")
    steering_vec = torch.randn(256)
    hook = SteeringHook(
        layer_idx=2,
        steering_vector=steering_vec,
        coefficient=1.0,
        position="all"
    )
    hook.attach(model.blocks[2])

    x = torch.randint(0, 100, (1, 10))
    with torch.no_grad():
        out_steered = model(x)
    print(f"    Output shape: {list(out_steered.shape)}")
    hook.remove()

    # Caching hook
    print("\n2.2 Caching hook:")
    cache_hook = CachingHook(layer_idx=3, max_cache_size=10)
    cache_hook.attach(model.blocks[3])

    with torch.no_grad():
        _ = model(x)

    cached = cache_hook.get_cached()
    print(f"    Cached activations shape: {list(cached.shape)}")
    cache_hook.remove()

    # Multiple hooks
    print("\n2.3 Multiple hooks:")
    hooks = [
        SteeringHook(layer_idx=1, steering_vector=torch.randn(256)),
        CachingHook(layer_idx=3),
    ]
    attach_hooks(model, hooks)

    with torch.no_grad():
        _ = model(x)
    print(f"    Active hooks: {len(hooks)}")
    print(f"    Cached shape: {list(hooks[1].get_cached().shape)}")

    for h in hooks:
        h.remove()


def demo_steerer():
    """Demonstrate the high-level steering interface."""
    print("\n" + "=" * 60)
    print("DEMO 3: ActivationSteerer")
    print("=" * 60)

    model = DemoTransformer(vocab_size=100, hidden_dim=256, num_layers=6)
    tokenizer = DemoTokenizer(vocab_size=100)
    steerer = ActivationSteerer(model, tokenizer)

    # Create steering vector
    vec = SteeringVector(
        vector=torch.randn(256),
        layer_idx=3,
        name="test_direction",
    )

    # Generate without steering
    print("\n3.1 Generation without steering:")
    result = steerer.generate(
        prompt="Hello world",
        vectors=None,
        max_new_tokens=10,
    )
    print(f"    Steering applied: {result.steering_applied}")
    print(f"    Output: {result.text[:30]}...")

    # Generate with steering
    print("\n3.2 Generation with steering:")
    result = steerer.generate(
        prompt="Hello world",
        vectors=vec,
        config=SteeringConfig(coefficient=1.5),
        max_new_tokens=10,
    )
    print(f"    Steering applied: {result.steering_applied}")
    print(f"    Coefficient: {result.config.coefficient}")

    # Compare steered vs unsteered
    print("\n3.3 Comparing steered vs unsteered:")
    result = steerer.generate(
        prompt="Test input",
        vectors=vec,
        max_new_tokens=10,
        compare_unsteered=True,
    )
    print(f"    Original output: {result.original_text[:20]}...")
    print(f"    Steered output: {result.text[:20]}...")

    # Coefficient sweep
    print("\n3.4 Coefficient sweep:")
    coefficients = [0.0, 0.5, 1.0, 2.0]
    results = steerer.sweep_coefficients(
        prompt="Test",
        vector=vec,
        coefficients=coefficients,
        max_new_tokens=5,
    )
    for r in results:
        print(f"    coeff={r.metadata['coefficient']:.1f}: {r.text[:15]}...")


def demo_analyzer():
    """Demonstrate activation analysis."""
    print("\n" + "=" * 60)
    print("DEMO 4: ActivationAnalyzer")
    print("=" * 60)

    model = DemoTransformer(vocab_size=100, hidden_dim=256, num_layers=6)
    tokenizer = DemoTokenizer(vocab_size=100)
    analyzer = ActivationAnalyzer(model, tokenizer)

    vec = SteeringVector(
        vector=torch.randn(256),
        layer_idx=3,
        name="analysis_vector",
    )

    # Projection analysis
    print("\n4.1 Projection analysis:")
    texts = ["Hello world", "Test input", "Another example"]
    analysis = analyzer.analyze_projection(texts, vec)
    print(f"    Mean projection: {analysis.mean_projection:.4f}")
    print(f"    Std projection: {analysis.std_projection:.4f}")
    print(f"    Range: [{analysis.min_projection:.4f}, {analysis.max_projection:.4f}]")

    # Compare projections
    print("\n4.2 Comparing positive vs negative examples:")
    positive = ["I am very happy", "This is wonderful", "Great success"]
    negative = ["I am very sad", "This is terrible", "Bad failure"]
    comparison = analyzer.compare_projections(positive, negative, vec)
    print(f"    Positive mean: {comparison['positive_mean']:.4f}")
    print(f"    Negative mean: {comparison['negative_mean']:.4f}")
    print(f"    Separation: {comparison['separation']:.4f}")
    print(f"    Effect size: {comparison['effect_size']:.4f}")

    # Steering effectiveness
    print("\n4.3 Measuring steering effectiveness:")
    effectiveness = analyzer.measure_steering_effect(
        text="Test sentence for analysis",
        vector=vec,
        coefficient=1.0,
    )
    print(f"    Projection shift: {effectiveness.projection_shift:.4f}")
    print(f"    Cosine similarity: {effectiveness.cosine_similarity:.4f}")
    print(f"    Magnitude ratio: {effectiveness.magnitude_ratio:.4f}")

    # Vector similarity
    print("\n4.4 Vector similarity:")
    vec2 = SteeringVector(torch.randn(256), layer_idx=3, name="vec2")
    sim = analyzer.vector_similarity(vec, vec2)
    print(f"    Random vectors: {sim:.4f}")
    same_sim = analyzer.vector_similarity(vec, vec)
    print(f"    Same vector: {same_sim:.4f}")


def demo_save_load():
    """Demonstrate saving and loading vectors."""
    print("\n" + "=" * 60)
    print("DEMO 5: Saving and Loading")
    print("=" * 60)

    import tempfile
    from pathlib import Path

    vec = SteeringVector(
        vector=torch.randn(256),
        layer_idx=5,
        name="persistent_vector",
        description="A vector that survives disk round-trip",
        metadata={"source": "demo", "version": 1},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_vector"

        # Save
        print("\n5.1 Saving vector:")
        save_steering_vector(vec, path)
        print(f"    Saved to: {path}")
        print(f"    Files created: .pt, .json")

        # Load
        print("\n5.2 Loading vector:")
        loaded = load_steering_vector(path)
        print(f"    Name: {loaded.name}")
        print(f"    Layer: {loaded.layer_idx}")
        print(f"    Dim: {loaded.dim}")

        # Verify
        print("\n5.3 Verifying integrity:")
        match = torch.allclose(vec.vector, loaded.vector)
        print(f"    Vectors match: {match}")


def demo_contrastive_pairs():
    """Show pre-defined contrastive pairs."""
    print("\n" + "=" * 60)
    print("DEMO 6: Pre-defined Contrastive Pairs")
    print("=" * 60)

    print("\n6.1 Honesty pairs:")
    for pair in HONESTY_PAIRS:
        print(f"    + {pair.positive[:40]}...")
        print(f"    - {pair.negative[:40]}...")
        print()

    print("6.2 Helpfulness pairs:")
    for pair in HELPFULNESS_PAIRS:
        print(f"    + {pair.positive[:40]}...")
        print(f"    - {pair.negative[:40]}...")
        print()


def main():
    print("\n" + "#" * 60)
    print("#" + " " * 18 + "ACTIVATION STEERING" + " " * 19 + "#")
    print("#" + " " * 12 + "Control Model Behavior at Runtime" + " " * 11 + "#")
    print("#" * 60)

    demo_steering_vector_operations()
    demo_hooks()
    demo_steerer()
    demo_analyzer()
    demo_save_load()
    demo_contrastive_pairs()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
