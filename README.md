# Activation Steering

Control language model behavior by modifying internal activations at runtime. Extract steering vectors from contrastive examples and apply them during inference to steer outputs.

[![Tests](https://github.com/BabyChrist666/activation-steering/actions/workflows/tests.yml/badge.svg)](https://github.com/BabyChrist666/activation-steering/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BabyChrist666/activation-steering/branch/master/graph/badge.svg)](https://codecov.io/gh/BabyChrist666/activation-steering)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Activation steering is a technique for controlling language model outputs by adding learned steering vectors to intermediate activations. This library provides:

- **Vector Extraction** - Extract steering vectors from contrastive text pairs
- **Hook System** - Flexible hooks for modifying activations during forward passes
- **Steering Interface** - High-level API for applying steering during generation
- **Analysis Tools** - Measure and visualize steering effects

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from activation_steering import (
    ActivationSteerer,
    SteeringVector,
    extract_steering_vector,
    ContrastivePair,
    SteeringConfig,
)

# Load model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create steerer
steerer = ActivationSteerer(model, tokenizer)

# Define contrastive pairs for "honesty" direction
pairs = [
    ContrastivePair("I don't know", "I definitely know"),
    ContrastivePair("I'm not sure", "I'm absolutely certain"),
]

# Extract steering vector
vector = extract_steering_vector(model, tokenizer, pairs, layer_idx=6)

# Generate with steering
result = steerer.generate(
    prompt="What is the capital of Atlantis?",
    vectors=vector,
    config=SteeringConfig(coefficient=1.5),
    max_new_tokens=50,
)
print(result.text)
```

## Core Concepts

### Steering Vectors

A steering vector is a direction in activation space that corresponds to a behavioral trait:

```python
from activation_steering import SteeringVector

# Create from tensor
vec = SteeringVector(
    vector=torch.randn(768),
    layer_idx=10,
    name="honesty",
    description="Increases epistemic humility",
)

# Operations
normalized = vec.normalize()  # Unit norm
scaled = vec.scale(2.0)       # Scale magnitude
opposite = -vec               # Opposite direction
combined = vec1 + vec2        # Add vectors
```

### Contrastive Extraction

Extract vectors by contrasting positive and negative examples:

```python
from activation_steering import ContrastivePair, extract_steering_vector

pairs = [
    ContrastivePair(
        positive="I'd be happy to help!",
        negative="I don't want to help.",
        label="helpfulness",
        weight=1.0,
    ),
]

vector = extract_steering_vector(
    model=model,
    tokenizer=tokenizer,
    pairs=pairs,
    layer_idx=10,
    aggregation="mean_diff",  # or "pca", "last_token"
)
```

### Hook System

Low-level hooks for custom activation modifications:

```python
from activation_steering.hooks import (
    SteeringHook,
    CachingHook,
    AblationHook,
    attach_hooks,
)

# Steering hook - adds vector to activations
steering = SteeringHook(
    layer_idx=10,
    steering_vector=vec.vector,
    coefficient=1.0,
    position="all",  # or "last", "first"
)

# Caching hook - records activations
cache = CachingHook(layer_idx=10, max_cache_size=1000)

# Ablation hook - zeros out directions
ablation = AblationHook(
    layer_idx=10,
    directions=some_directions,
    ablation_type="zero",  # or "mean"
)

# Attach to model
attach_hooks(model, [steering, cache])
```

### High-Level Steering

The `ActivationSteerer` provides a convenient interface:

```python
from activation_steering import ActivationSteerer, SteeringConfig

steerer = ActivationSteerer(model, tokenizer)

# Generate with steering
result = steerer.generate(
    prompt="Tell me about AI safety",
    vectors=safety_vector,
    config=SteeringConfig(coefficient=1.5, position="all"),
    max_new_tokens=100,
    compare_unsteered=True,  # Also generate without steering
)

print("Steered:", result.text)
print("Original:", result.original_text)

# Sweep coefficients to find optimal strength
results = steerer.sweep_coefficients(
    prompt="What is 2+2?",
    vector=honesty_vector,
    coefficients=[0.0, 0.5, 1.0, 1.5, 2.0],
)
```

### Analysis

Measure and understand steering effects:

```python
from activation_steering import ActivationAnalyzer

analyzer = ActivationAnalyzer(model, tokenizer)

# Compare projections on positive vs negative examples
comparison = analyzer.compare_projections(
    positive_texts=["I'm not sure...", "I might be wrong..."],
    negative_texts=["I'm certain!", "I definitely know!"],
    vector=honesty_vector,
)
print(f"Separation: {comparison['separation']:.3f}")
print(f"Effect size: {comparison['effect_size']:.3f}")

# Find best layer for steering
best_layer, score = analyzer.find_best_layer(
    pairs=contrastive_pairs,
    layer_range=(5, 15),
)

# Measure steering effectiveness
effectiveness = analyzer.measure_steering_effect(
    text="Test input",
    vector=my_vector,
    coefficient=1.0,
)
print(f"Projection shift: {effectiveness.projection_shift:.3f}")
```

## Pre-defined Steering Directions

```python
from activation_steering.vectors import (
    HONESTY_PAIRS,
    HELPFULNESS_PAIRS,
    SAFETY_PAIRS,
)

# Extract standard vectors
honesty_vec = extract_steering_vector(model, tokenizer, HONESTY_PAIRS, layer_idx=10)
helpful_vec = extract_steering_vector(model, tokenizer, HELPFULNESS_PAIRS, layer_idx=10)
```

## Saving and Loading Vectors

```python
from activation_steering import save_steering_vector, load_steering_vector

# Save
save_steering_vector(vector, "vectors/honesty.pt")

# Load
loaded = load_steering_vector("vectors/honesty.pt", device="cuda")
```

## Use Cases

- **Reducing Hallucinations** - Steer toward uncertainty and honesty
- **Increasing Helpfulness** - Boost cooperative behavior
- **Safety Research** - Study how behaviors are encoded
- **Interpretability** - Understand activation space structure
- **Alignment** - Fine-tune behavior without retraining

## References

- [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248)
- [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)

## License

MIT
