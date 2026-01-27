# Digital Primordial Soup

A GPU-accelerated artificial life simulation where neural network-driven organisms compete for survival through predation, reproduction, and evolution.

## Features

- **Neuroevolution**: Each organism has its own neural network that controls behavior. Weights are inherited and mutated during reproduction
- **Chemical Signaling**: Evolvable chemical system where species secrete and sense 4 different chemicals that diffuse through the environment
- **Dynamic Combat**: Success rates (10%-90%) determined by emergent chemical field strength, not hardcoded values
- **Reinforcement Learning**: Neural networks are fine-tuned within each generation based on survival rewards
- **Geographic Speciation**: Species split into two when geographically separated populations are detected (checked every 100 generations)
- **Dynamic Speciation**: When one species dominates (>75%), it automatically splits into a new species to maintain ecosystem diversity
- **GPU Acceleration**: All computations run on CUDA/MPS/CPU via PyTorch for real-time simulation
- **Network Persistence**: Best-performing neural networks are saved and loaded across sessions

## Demo

![Simulation Screenshot](docs/demo.png)

The visualization shows:
- **Top left**: Global population dynamics over all generations
- **Bottom left**: Real-time grid view of organisms (colors = species, brightness = energy)
- **Bottom middle**: Recent species population trends (last 100 generations)
- **Right panel**: Species legend with colored squares, population counts, and percentages

## Installation

```bash
# Clone the repository
git clone https://github.com/geyuxu/digital-primordial-soup.git
cd digital-primordial-soup

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run simulation (default 20x20 grid)
python main.py

# Run with larger grid
python main.py --grid 30

# Validation mode: test trained network vs random networks
python main.py --validate
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--grid`, `-g` | Grid size (default: 20) |
| `--validate`, `-v` | Validation mode: S0 uses trained weights, S1/S2 use random |

## How It Works

### Species & Predation
- All species start with identical parameters
- Differentiation emerges from neural network weights and evolvable chemical affinity
- Every species can prey on every other species
- **Dynamic combat success**: 10%-90% based on local chemical field strength (attacker vs defender)
- Successful hunts grant 120% of prey's energy

### Chemical Signaling System
- **4 Chemical Types**: Each species has evolvable affinity for secreting different chemicals
- **Diffusion**: Chemicals spread through the environment using GPU-accelerated convolution
- **Decay**: Chemicals naturally degrade over time (5% per step)
- **Emergent Behavior**: Chemical fields create "strength zones" that affect combat outcomes
- **Neural Sensing**: Networks receive chemical concentrations as additional inputs

### Neural Network
- **Input (24 neurons)**:
  - 8 neighbor energy levels
  - Same/different species counts
  - Own energy and neighbor density
  - 4 local chemical concentrations
  - 8 reserved slots for future features
- **Hidden (variable neurons)**: Fully connected with tanh activation
- **Output (7 actions)**: Stay, Move (4 directions), Eat, Reproduce

### Evolution Mechanisms
1. **Inheritance**: Offspring inherit parent's neural network and chemical affinity with small mutations
2. **RL Fine-tuning**: Successful actions (hunting, escaping, reproducing) reinforce neural network weights
3. **Dominance Speciation**: Dominant species (>75% population) split to prevent competitive exclusion
4. **Geographic Speciation**: Species split when separated into distinct spatial clusters (checked every 100 generations)

### Fitness & Persistence
- Fitness = lifetime + reproduction_count × 10
- Best network saved every 50 generations to `best_brain.pt`
- 20% of initial cells inherit saved weights on startup

## Configuration

Key parameters in `main.py`:

```python
# Simulation
GRID_SIZE = 20
INITIAL_NUM_SPECIES = 3
MAX_SPECIES = 10

# Chemical Signaling
NUM_CHEMICALS = 4
CHEMICAL_DIFFUSION = 0.3
CHEMICAL_DECAY = 0.05
CHEMICAL_SECRETION = 0.1

# Evolution
MUTATION_RATE = 0.1
DOMINANCE_THRESHOLD = 0.75
SPLIT_MUTATION_RATE = 0.3
BLOB_CHECK_INTERVAL = 100

# Reinforcement Learning
RL_LEARNING_RATE = 0.01
REWARD_EAT_PREY = 2.0
REWARD_SURVIVE_ATTACK = 1.0
REWARD_REPRODUCE = 1.5
```

## Project Structure

```
digital-primordial-soup/
├── main.py           # Main simulation code
├── best_brain.pt     # Saved neural network weights
├── requirements.txt  # Python dependencies
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch (with CUDA/MPS support recommended)
- NumPy
- Matplotlib
- SciPy (for geographic speciation blob detection)

## Version History

### v1.1-chemical-signaling (current)
- Chemical signaling system with 4 evolvable chemical types
- Dynamic combat (10%-90% success based on chemical field strength)
- Geographic speciation via blob detection (every 100 generations)
- 3-panel UI with colored species legend
- INPUT_SIZE expanded to 24 (added chemical inputs)

### v1.0-single-cell
- Basic single-cell neuroevolution
- Fixed 50% combat success rate
- Dominance-based speciation only
- 2-panel UI layout

## License

MIT License

---

# 数字原始汤

一个 GPU 加速的人工生命模拟器，神经网络驱动的生物通过捕食、繁殖和进化竞争生存。

## 特性

- **神经进化**：每个生物都有自己的神经网络控制行为，权重在繁殖时遗传和变异
- **化学信号系统**：可进化的化学系统，物种分泌和感知 4 种不同的化学物质，在环境中扩散
- **动态战斗**：成功率（10%-90%）由涌现的化学场强度决定，而非硬编码数值
- **强化学习**：神经网络根据生存奖励在每一代内进行微调
- **地理物种形成**：当检测到地理分离的种群时物种分裂（每 100 代检查一次）
- **优势物种形成**：当某物种占比超过 75% 时自动分裂，维持生态系统多样性
- **GPU 加速**：所有计算通过 PyTorch 在 CUDA/MPS/CPU 上运行
- **网络持久化**：最优神经网络跨会话保存和加载

## 运行方式

```bash
# 默认模式
python main.py

# 大网格
python main.py --grid 30

# 验证模式：测试训练网络 vs 随机网络
python main.py --validate
```

## 工作原理

### 物种与捕食
- 所有物种参数完全相同
- 差异来自神经网络权重和可进化的化学亲和力
- 任意物种可捕食其他物种
- **动态战斗成功率**：10%-90%，基于局部化学场强度（攻击者 vs 防御者）

### 化学信号系统
- **4 种化学类型**：每个物种对分泌不同化学物质有可进化的亲和力
- **扩散**：化学物质通过 GPU 加速卷积在环境中扩散
- **衰减**：化学物质自然降解（每步 5%）
- **涌现行为**：化学场创建影响战斗结果的"强度区域"
- **神经感知**：神经网络接收化学浓度作为额外输入

### 进化机制
1. **遗传**：后代继承父代神经网络和化学亲和力 + 小变异
2. **强化学习**：成功行为（捕猎、逃脱、繁殖）强化神经网络
3. **优势物种分裂**：占比 >75% 的物种自动分裂防止竞争排斥
4. **地理物种分裂**：当物种分离成不同空间集群时分裂（每 100 代检查）

### 适应度与持久化
- 适应度 = 存活代数 + 繁殖次数 × 10
- 每 50 代保存最优网络到 `best_brain.pt`
- 启动时 20% 的细胞继承保存的权重
