<div align="center">

# Yuxu's Game of Life

### A GPU-Accelerated Artificial Life Simulation with Emergent Neural Evolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18397035.svg)](https://doi.org/10.5281/zenodo.18397035)

*Extending Conway's Game of Life into a massively parallel neuroevolution ecosystem where thousands of neural agents compete, reproduce, and evolve in real-time.*

![Simulation Demo](docs/demo.gif)

**[Watch Full Demo on YouTube](https://youtu.be/CeMnTrSrS8k)** | **[Read the Blog Post](https://yuxu.ge/blog/post.html?p=posts/2026/neuroevo-life)** | **[Read the Paper](#citation)**

</div>

---

## Abstract

This project implements a GPU-accelerated Artificial Life (ALife) simulation that extends the classical cellular automaton paradigm into a continuous neuroevolution ecosystem. Each cell is governed by an individual neural network whose weights are subject to both **within-lifetime reinforcement learning** (policy gradient) and **cross-generational neuroevolution** (mutation and crossover). A novel **12-dimensional genomic system** encodes both neural phenotypes and chemical signaling affinities, enabling emergent speciation without predefined species boundaries.

**Key Finding**: In validation experiments, a trained "super-lineage" starting from just 1% of the population achieved **100% ecosystem dominance** within ~5000 generations, demonstrating the compound advantage of hybrid RL-neuroevolution architectures.

---

## Table of Contents

- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [The 12-Dimensional Genome](#the-12-dimensional-genome)
- [GPU Vectorization Strategy](#gpu-vectorization-strategy)
- [Hybrid Evolution Mechanism](#hybrid-evolution-mechanism)
- [Experimental Results](#experimental-results)
- [Configuration](#configuration)
- [Controls](#controls)
- [A/B Testing Framework](#ab-testing-framework)
- [Performance Benchmarks](#performance-benchmarks)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Key Features

### Massive GPU Parallelization
- **10,000+ simultaneous neural agents** evaluated in parallel via PyTorch
- Vectorized operations using `torch.einsum` for batched forward passes
- Chemical diffusion computed via `F.conv2d` with circular padding
- Pre-allocated tensor buffers eliminate runtime memory allocation overhead

### Hybrid Evolution Architecture
- **Reinforcement Learning**: Policy gradient updates with advantage baseline for within-lifetime adaptation
- **Neuroevolution**: Sexual reproduction with 50/50 neural crossover and Gaussian mutation
- **Dual pressure**: Organisms must both learn quickly (RL) and inherit good priors (evolution)

### 12-Dimensional Genomic Identity
- **8D Neural Fingerprint**: Statistical summary of network weights (genotype)
- **4D Chemical Affinity**: Secretion/detection preferences for signaling (phenotype)
- Cosine similarity-based compatibility for mating and predation decisions

### Emergent Ecological Dynamics
- **No predefined species**: Clusters emerge naturally from genome similarity
- **Chemical ecology**: 4-chemical signaling system with diffusion and decay
- **Tissue mechanics**: Enclosed spaces fill with averaged neighbor genomes

---

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ (with CUDA or MPS support recommended)
- 4GB+ GPU memory (for 100x100 grid)

### Installation

```bash
# Clone the repository
git clone https://github.com/geyuxu/yuxus-life-of-game.git
cd yuxus-life-of-game

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Launch Simulation

```bash
# Auto-detects GPU: CUDA > MPS > CPU
python main.py

# Specify grid size
python main.py --grid 50
```

### Device Support

| Platform | Device | Command |
|----------|--------|---------|
| NVIDIA GPU | CUDA | Auto-detected |
| Apple Silicon | MPS | Auto-detected |
| CPU Fallback | CPU | Auto-detected |

The simulation automatically selects the optimal compute device:

```python
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SIMULATION LOOP (per step)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Sensing    │───▶│   Thinking   │───▶│   Acting     │              │
│  │  _build_     │    │  _batch_     │    │  _execute_   │              │
│  │   inputs()   │    │  forward()   │    │   actions()  │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ 24D Input    │    │ einsum for   │    │ Movement,    │              │
│  │ Tensor       │    │ 10K agents   │    │ Eating,      │              │
│  │ [H,W,24]     │    │ in parallel  │    │ Reproduction │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    LEARNING & EVOLUTION                          │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │  Reinforcement Learning          │  Neuroevolution               │  │
│  │  ─────────────────────────       │  ─────────────────────────    │  │
│  │  • Policy gradient updates       │  • Sexual crossover (50/50)   │  │
│  │  • Advantage baseline            │  • Gaussian mutation (σ=0.1)  │  │
│  │  • Experience replay buffer      │  • Genome inheritance         │  │
│  │  • Rewards: eat, escape, repro   │  • Fitness-based selection    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    CHEMICAL ECOLOGY                              │  │
│  │  • 4-chemical signaling field [4, H, W]                          │  │
│  │  • Diffusion via conv2d (rate=0.3)                               │  │
│  │  • Decay per step (rate=0.05)                                    │  │
│  │  • Genome-driven secretion patterns                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The 12-Dimensional Genome

Each organism carries a **12-dimensional genome vector** that serves as its genetic identity:

```
Genome[12] = [Neural Fingerprint (8D)] + [Chemical Affinity (4D)]
             ├── Genotype ──────────┤   ├── Phenotype ─────────┤
```

### Neural Fingerprint (Dimensions 0-7)

Statistical summary extracted from network weights, capturing the "shape" of the neural architecture:

| Dim | Description | Source |
|-----|-------------|--------|
| 0-3 | mean, std, abs_mean, max | Input→Hidden weights (W1) |
| 4-7 | mean, std, abs_mean, max | Hidden→Output weights (W2) |

### Chemical Affinity (Dimensions 8-11)

Encodes preferences for the 4-chemical signaling system:

| Dim | Chemical | Function |
|-----|----------|----------|
| 8 | α (Alpha) | Territory marking |
| 9 | β (Beta) | Aggregation signal |
| 10 | γ (Gamma) | Danger warning |
| 11 | δ (Delta) | Resource indicator |

### Genome-Based Interactions

Compatibility is determined by **cosine similarity** in the chemical affinity subspace:

```python
# Mating: requires high similarity (cosine > 0.8)
cosine_sim = F.cosine_similarity(genome1[8:12], genome2[8:12])
can_mate = cosine_sim > 0.8

# Predation: requires low similarity (cosine ≤ 0.8)
can_eat = cosine_sim <= 0.8
```

This creates natural species boundaries without explicit species definitions.

---

## GPU Vectorization Strategy

### The Challenge

Simulating 10,000 neural agents each with:
- 24 input neurons
- 8 hidden neurons (expandable to 240)
- 7 output actions

Traditional loop-based approach: **O(N × forward_pass_time)** — impossibly slow.

### The Solution: Batched Tensor Operations

**1. Batched Forward Pass via `einsum`**

```python
def _batch_forward(self, inputs):
    # inputs: [H, W, 24] - all agents' sensory data
    # w1: [H, W, 24, 240] - all agents' input weights
    # w2: [H, W, 240, 7] - all agents' output weights

    # Single einsum computes ALL hidden activations
    h = torch.tanh(torch.einsum('ijk,ijkl->ijl', inputs, self.w1))

    # Single einsum computes ALL action logits
    logits = torch.einsum('ijk,ijkl->ijl', h, self.w2)

    return F.softmax(logits, dim=-1)  # [H, W, 7]
```

**2. Chemical Diffusion via `conv2d`**

```python
def _diffuse_chemicals(self):
    kernel = torch.tensor([[
        [0.05, 0.1, 0.05],
        [0.1,  0.4, 0.1],
        [0.05, 0.1, 0.05]
    ]])  # Gaussian-like diffusion

    for chem_id in range(4):
        padded = F.pad(self.chemicals[chem_id], (1,1,1,1), mode='circular')
        self.chemicals[chem_id] = F.conv2d(padded, kernel)
```

**3. Pre-Allocated Buffers**

```python
# In __init__: allocate once
self._input_buffer = torch.zeros((size, size, INPUT_SIZE), device=DEVICE)
self._neighbor_buffer = torch.zeros((size, size, 8), device=DEVICE)

# In _build_inputs: reuse via slice assignment
for i, (dr, dc) in enumerate(dirs):
    self._neighbor_buffer[:, :, i] = self._get_shifted(energy, dr, dc)
```

---

## Hybrid Evolution Mechanism

### Within-Lifetime: Reinforcement Learning

**Policy Gradient with Advantage Baseline**

```python
# Compute advantage (Optimization A: stability improvement)
advantages = rewards - self.reward_baseline  # Subtracting baseline
self.reward_baseline = 0.95 * self.reward_baseline + 0.05 * rewards.mean()

# Update weights proportional to advantage
delta_w2 = LR * advantages * hidden * action_onehot
```

**Reward Structure**:
| Event | Reward |
|-------|--------|
| Successful hunt | +2.0 |
| Escape attack | +1.0 |
| Reproduction | +1.5 |

### Cross-Generational: Neuroevolution

**Sexual Reproduction with Crossover**

```python
# 50/50 random selection of neurons from each parent
crossover_mask = torch.rand(shape) > 0.5
child_weights = torch.where(crossover_mask, parent1_weights, parent2_weights)

# Gaussian mutation
child_weights += torch.randn_like(child_weights) * MUTATION_RATE
```

**Fitness Function** (Multi-objective):

```
Fitness = Lifetime × 1.0
        + Reproduction × 10.0
        + Diversity × 5.0
        + Energy × 0.5
```

---

## Experimental Results

### Super-Lineage Emergence

Starting conditions:
- **1% trained lineage** (inheriting saved weights)
- **99% random lineage** (random initialization)

| Generation | Trained % | Random % | Observation |
|------------|-----------|----------|-------------|
| 0 | 1% | 99% | Initial seeding |
| 500 | 15% | 85% | Rapid expansion begins |
| 2000 | 60% | 40% | Majority achieved |
| 5000 | 100% | 0% | **Complete dominance** |

### Validation Metrics

```
======================================================================
GENERATIONAL VALIDATION - Generation 5000
======================================================================

Population by Lineage:
  Trained Lineage:              10000 (100.0%)
  Random (No trained ancestry):     0 (  0.0%)

Performance Comparison (Trained vs Random baseline):
  Lifetime:     1.45x better
  Reproduction: 1.62x better
  Energy:       1.28x better

Conclusion: Training provides compound evolutionary advantage
======================================================================
```

---

## Configuration

All parameters are centralized in `config.py`:

```python
# Core Simulation
GRID_SIZE = 100              # World dimensions
MAX_ENERGY = 100.0           # Energy cap

# Neural Architecture
INPUT_SIZE = 24              # Sensory inputs
SPECIES_HIDDEN_SIZE = 8      # Hidden neurons
NUM_ACTIONS = 7              # Possible actions

# Evolution
MUTATION_RATE = 0.1          # Weight mutation σ
MATE_GENOME_THRESHOLD = 0.5  # Compatibility threshold (legacy)

# Reinforcement Learning
RL_LEARNING_RATE = 0.01      # Policy gradient LR
RL_USE_REPLAY = True         # Experience replay buffer

# Chemical System
NUM_CHEMICALS = 4            # Signaling channels
CHEMICAL_DIFFUSION = 0.3     # Spread rate
CHEMICAL_DECAY = 0.05        # Decay rate
```

View current configuration:
```bash
python -c "from config import print_config; print_config()"
```

---

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume simulation |
| `R` | Reset simulation |
| `S` | Save best network |
| `C` | Toggle chemical overlay |
| `G` | Toggle grid lines |
| `H` | Toggle genome heatmap |
| `+`/`-` | Adjust simulation speed |
| `1-4` | Select parameter to adjust |
| `↑`/`↓` | Modify selected parameter |
| `Scroll` | Zoom in/out |
| `Drag` | Pan camera |
| `Ctrl+S` | Save checkpoint |
| `Ctrl+L` | Load checkpoint |
| `ESC` | Quit |

---

## A/B Testing Framework

Compare configurations scientifically:

```bash
python ab_test.py --generations 1000 --output results.json
```

Example comparison:
```python
config_a = {'MUTATION_RATE': 0.2, 'RL_LEARNING_RATE': 0.005}
config_b = {'MUTATION_RATE': 0.05, 'RL_LEARNING_RATE': 0.02}
```

**Tracked Metrics**: Population, Fitness, Diversity, Trained Ratio, Energy, Lifetime, Reproduction

---

## Performance Benchmarks

| Metric | Before Optimization | After Optimization | Improvement |
|--------|---------------------|-------------------|-------------|
| Step Time | 180ms | 40ms | **4.5x** |
| FPS | 10 | 50+ | **5x** |
| GPU Memory | Dynamic | Pre-allocated | Stable |
| Population Cap | 5,000 | 10,000+ | **2x** |

**Hardware**: NVIDIA RTX 3080 / Apple M1 Pro

---

## Project Structure

```
yuxus-life-of-game/
├── main.py              # Entry point & Pygame renderer
├── evolution.py         # Core simulation engine (GPULifeGame)
├── config.py            # All configurable parameters
├── ab_test.py           # A/B testing framework
├── requirements.txt     # Python dependencies
├── best_brain.pt        # Saved neural network (auto-generated)
├── checkpoint.pt        # Full simulation state (Ctrl+S)
├── docs/
│   └── demo.gif         # Demo animation
└── README.md
```

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{xu2025yuxus_game_of_life,
  author       = {Xu, Yuxu},
  title        = {Yuxu's Game of Life: A GPU-Accelerated Neuroevolution Ecosystem},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18397035},
  url          = {https://doi.org/10.5281/zenodo.18397035}
}
```

### Academic Context

This project was developed following the completion of the first semester of an **MSc in Artificial Intelligence** at the **University of York**, bridging 10+ years of senior software architecture experience with formal AI research methodologies. It demonstrates the practical application of:

- Reinforcement Learning (Policy Gradient Methods)
- Neuroevolution (NEAT-inspired crossover)
- GPU Computing (PyTorch Vectorization)
- Artificial Life (Cellular Automata Extensions)

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with PyTorch** | **Accelerated by CUDA/MPS** | **Inspired by Conway**

*"Life finds a way." — but with gradient descent.*

</div>
