# Yuxu's Game of Life

A GPU-accelerated genome-based evolution simulation where neural network organisms compete, reproduce, and evolve through natural selection and reinforcement learning.

## Features

### Core Evolution System
- **Genome-Based Identity**: 12-dimensional genome (8 neural fingerprint + 4 chemical affinity) determines each organism's identity
- **No Fixed Species**: Organisms cluster naturally based on genome similarity, no predetermined species
- **Sexual Reproduction**: Genetic crossover where offspring inherit 50% neurons from each parent
- **Emergent Speciation**: Similar genomes form natural breeding groups, visible through color clustering

### Advanced Training & Validation
- **Continuous Evolution**: Networks evolve through both inheritance and reinforcement learning
- **Lineage Tracking**: Validates training effectiveness by tracking trained vs random lineages across generations
- **Auto-Save System**: Best networks automatically saved every 2 minutes (configurable)
- **Generational Metrics**: Track performance of Gen0, Gen1-5, Gen6+ descendants vs pure random

### Chemical Ecology
- **4-Chemical System**: Genome-encoded chemical preferences for signaling and territory marking
- **Dynamic Diffusion**: Chemicals spread (30%) and decay (5%) each step
- **Genome-Driven Secretion**: Each cell secretes based on its chemical affinity genes

### Technical Excellence
- **GPU Acceleration**: All computations via PyTorch (CUDA/MPS/CPU) for real-time performance
- **Reinforcement Learning**: Policy gradient updates within each organism's lifetime
- **Smooth Rendering**: 2x supersampling anti-aliasing for high-quality visualization
- **Real-time Clustering**: Genome similarity analysis shows emergent groups

## Demo

![Simulation Screenshot](docs/screenshot.png)

**Visualization Elements**:
- **Main Grid**: Organisms as colored cells (color from genome fingerprint, brightness from energy)
- **Right Panel**:
  - Generation and population statistics
  - FPS and performance metrics
  - **Lineage Tracking**: Trained vs Random lineage comparison
  - **Genome Clusters**: Top groups by population with percentages

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/yuxus-game-of-life.git
cd yuxus-game-of-life

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Launch simulation
python3 main.py

# Controls:
#   SPACE  - Pause/Resume
#   R      - Reset simulation
#   S      - Save best network manually
#   C      - Toggle chemical overlay
#   G      - Toggle grid
#   +/-    - Adjust simulation speed
#   Wheel  - Zoom in/out
#   Drag   - Pan camera
#   ESC    - Quit
```

## How It Works

### Genome-Based Identity

Each organism has a **12-dimensional genome**:

```
Genome = [Neural Fingerprint (8-dim)] + [Chemical Affinity (4-dim)]

Neural Fingerprint:
  - Mean, Std, Abs-mean, Max from w1 layer
  - Mean, Std, Abs-mean, Max from w2 layer

Chemical Affinity:
  - Preferences for secreting 4 different chemicals
```

**Key Properties**:
- Genome distance < 0.5 → Compatible mates (can reproduce)
- Genome distance ≥ 0.5 → Valid prey (can be eaten)
- Similar genomes → Similar colors (natural clustering)

### Reproduction & Inheritance

**Sexual Reproduction** (preferred):
1. Find nearby organism with similar genome (distance < 0.5)
2. Offspring inherits:
   - 50% neurons from parent 1 (random selection)
   - 50% neurons from parent 2 (random selection)
   - Genome crosses over and mutates (10% rate)
3. Child placed in adjacent empty cell

**Asexual Reproduction** (fallback):
- If no compatible mate nearby, clone with mutation
- Ensures reproduction even when isolated

### Lineage Tracking System

The simulation tracks **trained lineages** to validate network quality:

```
Generation 0:
  - 1% cells inherit best_brain.pt (ELITE_RATIO=0.01)
  - 99% cells start with random weights

All cells continuously learn via reinforcement learning

Lineage Tracking:
  - Gen 0: Direct inheritance from saved weights
  - Gen 1-5: Near descendants (1-5 generations from trained)
  - Gen 6+: Far descendants (6+ generations from trained)
  - Random: No trained ancestry

Validation Metrics:
  - Population ratio (Trained vs Random)
  - Average lifetime comparison
  - Average reproduction comparison
  - Performance ratio (should be >1.0 if training works)
```

**How to Validate Training**:
- Set `ELITE_RATIO = 0.01` (1% trained, 99% random)
- Run for 5-10 minutes
- If trained lineage grows from 1% to >30%: ✓ Training successful
- If trained lineage stays at 1-2%: ✗ Training ineffective

### Neural Network Architecture

**Input (24 neurons)**:
- 8 neighbor energy levels
- Similar/different genome counts
- Own energy and neighbor density
- 4 local chemical concentrations
- 8 reserved slots

**Hidden (8 neurons)**: Fully connected with tanh activation

**Output (7 actions)**:
- Stay (0)
- Move: Up, Down, Left, Right (1-4)
- Eat (5)
- Reproduce (6)

### Reinforcement Learning

**Policy Gradient Updates**:
```python
Rewards:
  - Eat prey: +2.0
  - Escape attack: +1.0
  - Reproduce: +1.5

Learning rate: 0.01
Updates: Every step based on actions taken
```

All organisms learn continuously, creating an arms race between trained and random lineages.

### Tissue Mechanics

**Enclosed Space Filling**:
- Empty cells surrounded by 8 similar organisms get filled
- New cell averages genome and weights from neighbors
- Creates dense tissue structures

**Tissue Fission** (rare):
- Large connected organisms (100+ cells) may split
- 10% chance per check
- Mutation applied during split (50% keep, 50% randomize)

### Fitness & Persistence

**Fitness Function**:
```python
fitness = lifetime + reproduction_count × 10
```
(Reproduction weighted 10x because it's harder to achieve)

**Auto-Save**:
- Best network saved every 120 seconds (2 minutes)
- Asynchronous saving prevents stuttering
- Tracks best individual across entire population

## Configuration

All parameters in `config.py`:

```python
# Grid & Energy
GRID_SIZE = 100
INITIAL_ENERGY = 30.0
MAX_ENERGY = 100.0

# Evolution
MUTATION_RATE = 0.1
MATE_GENOME_THRESHOLD = 0.5  # Genome distance for mating

# Validation
ELITE_RATIO = 0.01  # 1% trained, 99% random
AUTO_SAVE_ENABLED = True
SAVE_INTERVAL_SECONDS = 120  # Auto-save every 2 minutes

# Chemical System
NUM_CHEMICALS = 4
CHEMICAL_DIFFUSION = 0.3
CHEMICAL_DECAY = 0.05

# Reinforcement Learning
RL_LEARNING_RATE = 0.01
REWARD_EAT_PREY = 2.0
REWARD_SURVIVE_ATTACK = 1.0
REWARD_REPRODUCE = 1.5

# Visualization
GENOME_BASED_COLOR = True  # Color by genome similarity
```

Run `python3 -c "from config import print_config; print_config()"` to see all settings.

## Project Structure

```
yuxus-game-of-life/
├── config.py        # All configurable parameters
├── evolution.py     # Core evolution engine (GPULifeGame class)
├── main.py          # Pygame renderer (entry point)
├── ab_test.py       # A/B testing framework
├── best_brain.pt    # Saved neural network weights (auto-generated)
├── checkpoint.pt    # Complete simulation checkpoint (Ctrl+S to save)
└── README.md
```

## A/B Testing Framework

Compare two different configurations side-by-side to find optimal parameters:

```bash
# Run A/B test comparing mutation rate vs RL learning rate
python3 ab_test.py --generations 1000 --output results.json

# Example output:
# Gen  100 | A: Pop=5234 Fit= 234.1 Div=1.42 | B: Pop=5891 Fit= 289.3 Div=1.38
# Gen  200 | A: Pop=6012 Fit= 341.2 Div=1.35 | B: Pop=7234 Fit= 412.5 Div=1.29
```

**Built-in Test**: High Mutation + Low RL vs Low Mutation + High RL

**Customize your own test** by editing `ab_test.py`:

```python
config_a = {
    'MUTATION_RATE': 0.2,
    'RL_LEARNING_RATE': 0.005,
}

config_b = {
    'MUTATION_RATE': 0.05,
    'RL_LEARNING_RATE': 0.02,
}
```

**Metrics Tracked**:
- Population size
- Average fitness (multi-objective)
- Genetic diversity
- Trained lineage ratio
- Average energy, lifetime, reproduction

Results saved to JSON with full metric history for later analysis.

## Understanding the Validation Results

### Example: Successful Training

```
Generation 0: Trained 1%, Random 99%
1 minute:     Trained 15%, Random 85%  ← Rapid growth = good training
5 minutes:    Trained 60%, Random 40%
10 minutes:   Trained 85%, Random 15%

Validation Report:
  Trained Lineage: 85%
  Random: 15%
  Lifetime Ratio: 1.3x ✓ BETTER

→ Conclusion: Training very successful!
```

### Example: Ineffective Training

```
Generation 0: Trained 1%, Random 99%
10 minutes:   Trained 1%, Random 99%  ← No growth = poor training

Validation Report:
  Trained Lineage: 1%
  Random: 99%
  Lifetime Ratio: 0.9x ✗ WORSE

→ Conclusion: Training did not learn useful behaviors
```

### Long-term Dynamics

After 5000+ generations, both trained and random lineages converge to optimal strategies through continuous reinforcement learning. At this point:
- Performance becomes similar (ratio ≈ 1.0)
- Population ratio stabilizes based on historical advantage
- Trained lineage maintains dominance due to early head-start

This validates that: (1) training provides early advantage, (2) RL allows random to catch up, (3) early advantage determines long-term dominance.

## Advanced Features

### Dominance-Based Mutation

When a genome cluster exceeds 30% of population (checked every 100 generations):
- System applies forced mutation to maintain diversity
- 70% of neurons randomized, 30% kept
- Prevents single lineage from complete domination
- Maintains evolutionary pressure

### Generational Validation Report

Every 200 generations, detailed console output:

```
======================================================================
GENERATIONAL VALIDATION - Generation 200
======================================================================

Population by Lineage:
  Gen 0 (Direct trained):       120 (  1.4%)
  Gen 1-5 (Near descendants):  2500 ( 29.1%)
  Gen 6+ (Far descendants):    4800 ( 55.9%)
  Random (No trained ancestry): 1180 ( 13.7%)
  ----------------------------------------
  Total Trained Lineage:        7420 ( 86.3%)

** PRIMARY COMPARISON: Trained Lineage vs Random **
  Lifetime:     58.3 vs  45.2  =  1.29x ✓ BETTER
  Reproduction:  2.42 vs  1.85  =  1.31x ✓ BETTER
  Energy:       27.8 vs  24.3  =  1.14x ✓ BETTER

Breakdown by Generation:
  Lifetime:     Gen0= 82.3  Gen1-5= 65.2  Gen6+= 58.1
  Reproduction: Gen0= 3.20  Gen1-5= 2.80  Gen6+= 2.50
  Energy:       Gen0= 32.1  Gen1-5= 29.4  Gen6+= 27.8
======================================================================
```

## Performance

- **Real-time simulation**: 60+ FPS on modern GPUs
- **Large populations**: Handles 5000-10000 organisms simultaneously
- **GPU acceleration**: 100x faster than CPU-only
- **Smooth rendering**: 2x supersampling for anti-aliased visuals

## Requirements

- Python 3.8+
- PyTorch (CUDA/MPS recommended for GPU acceleration)
- NumPy
- SciPy
- Pygame

Install via: `pip install -r requirements.txt`

## Tips & Tricks

### Faster Evolution
```python
# config.py
SPECIES_METABOLISM = 0.15  # Faster turnover
SPECIES_REPRO_THRESHOLD = 15  # Easier reproduction
```

### Longer Lifespans
```python
# config.py
SPECIES_METABOLISM = 0.05  # Slower metabolism
MAX_ENERGY = 150.0  # Higher energy cap
```

### Stricter Validation
```python
# config.py
ELITE_RATIO = 0.005  # Only 0.5% trained (very strict)
```

### More Diversity
```python
# config.py
DOMINANCE_THRESHOLD = 0.2  # Trigger mutation at 20% (earlier)
MUTATION_RATE = 0.15  # Higher mutation rate
```

## License

MIT License - See LICENSE file for details

---

# Yuxu的生命游戏

GPU加速的基因组进化模拟，神经网络驱动的生物通过自然选择和强化学习竞争、繁殖和进化。

## 核心特性

- **基因组身份系统**: 12维基因组（8维神经指纹 + 4维化学亲和力）决定生物身份
- **无固定物种**: 基于基因组相似性自然聚类，无预设物种
- **有性繁殖**: 后代从父母各继承50%神经元的遗传交叉
- **血统追踪**: 验证训练效果，追踪训练血统vs随机血统的多代表现
- **持续进化**: 通过遗传和强化学习双重机制持续优化
- **GPU加速**: PyTorch实现，支持CUDA/MPS/CPU实时运行

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行模拟
python3 main.py

# 控制:
#   空格 - 暂停/继续
#   R - 重置
#   S - 手动保存网络
#   +/- - 调整速度
```

## 训练验证方法

设置 `ELITE_RATIO = 0.01` (1%训练，99%随机)，运行5-10分钟：

- ✅ 训练血统从1%增长到>30%：训练成功
- ✗ 训练血统保持在1-2%：训练无效

通过这种严格的1 vs 99竞争验证训练网络的真实优势。

## 配置

所有参数在 `config.py` 中，可自由调整网格大小、突变率、学习率等。

运行 `python3 -c "from config import print_config; print_config()"` 查看当前配置。
