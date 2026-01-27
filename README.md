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

### Option 1: Matplotlib Renderer (Default)

```bash
# Run simulation with matplotlib (slower, good for analysis)
python main.py

# Run with larger grid
python main.py --grid 30

# Validation mode: test trained network vs random networks
python main.py --validate
```

### Option 2: Web Interface (Recommended)

```bash
# Start web interface with real-time chemical visualization
python web_interface.py

# Open browser to http://localhost:8050
# Features:
# - Real-time population charts
# - 4 chemical field heatmaps
# - Species statistics table
# - Pause/Resume/Reset controls
```

### Option 3: Pygame Renderer (Fastest)

```bash
# High-performance renderer with 60+ FPS
python pygame_renderer.py

# Keyboard controls:
# - SPACE: Pause/Resume
# - R: Reset simulation
# - S: Save best network manually
# - C: Toggle chemical overlay
# - G: Toggle grid
# - +/-: Adjust speed (1x to 10x)
# - Mouse wheel: Zoom
# - Click+Drag: Pan camera
```

### Command Line Options (main.py only)

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

### Genome-Based Visualization (Scheme C: Hybrid Genome)
- **12-Dimensional Genome**: Each organism has a unique genome vector combining:
  - Neural fingerprint (8-dim): Statistical features extracted from neural network weights (mean, std, abs_mean, max for each layer)
  - Chemical affinity (4-dim): Evolvable preferences for secreting different chemicals
- **Genome-to-Color Mapping**: Species colors are dynamically generated from their average genome
  - Similar genomes produce similar colors in HSV space
  - Color evolves continuously as neural networks and chemical preferences evolve
  - Creates intuitive visual clustering of genetically related species
- **Soft Transition**: Genome system coexists with discrete species IDs
  - Species IDs still used for game mechanics (backwards compatible)
  - Genome used only for visualization (can be toggled on/off)
  - Updates every 50 generations to reflect evolutionary changes

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
1. **Sexual Reproduction with Genetic Crossover**:
   - Organisms find mates (same-species neighbors)
   - Offspring inherits 50% neurons from each parent (interleaved crossover)
   - Falls back to asexual cloning if no mate available
   - Mutation applied after inheritance (10% rate)
2. **RL Fine-tuning**: Successful actions (hunting, escaping, reproducing) reinforce neural network weights
3. **Dominance Speciation**: Dominant species (>75% population) split to prevent competitive exclusion
4. **Geographic Speciation**: Species split when separated into distinct spatial clusters (checked every 100 generations)

### Fitness & Persistence
- Fitness = lifetime + reproduction_count × 10
- Best network saved every 50 generations to `best_brain.pt`
- **Asynchronous saving**: Uses background thread to avoid stuttering during save operations
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

# Genome-based Visualization
GENOME_BASED_COLOR = True
GENOME_COLOR_UPDATE_INTERVAL = 50

# Reinforcement Learning
RL_LEARNING_RATE = 0.01
REWARD_EAT_PREY = 2.0
REWARD_SURVIVE_ATTACK = 1.0
REWARD_REPRODUCE = 1.5
```

## Project Structure

```
digital-primordial-soup/
├── main.py              # Main simulation code (matplotlib renderer)
├── web_interface.py     # Web-based dashboard (Plotly Dash)
├── pygame_renderer.py   # High-performance renderer (Pygame)
├── best_brain.pt        # Saved neural network weights
├── requirements.txt     # Python dependencies
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch (with CUDA/MPS support recommended)
- NumPy
- Matplotlib
- SciPy (for geographic speciation blob detection)
- Plotly + Dash (for web interface)
- Pygame (for high-performance renderer)

## Version History

### v1.5-sexual-reproduction (current)
- Sexual reproduction with genetic crossover
- Mate finding: organisms search for same-species neighbors
- Genetic crossover: offspring inherits 50% neurons from each parent (interleaved)
- Fallback to asexual reproduction if no mate found
- Enhanced diversity: saved weights only 50% loaded, 50% random
- Chemical-influenced color mapping for more diverse species colors

### v1.4-genome-visualization
- Genome-based color system (Scheme C: Hybrid Genome)
- 12-dimensional genome: neural fingerprint (8-dim) + chemical affinity (4-dim)
- Dynamic genome-to-color mapping in HSV space
- Species colors evolve continuously based on neural network and chemical evolution
- Soft transition: genome visualization coexists with discrete species IDs

### v1.3-async-saving
- Asynchronous model saving to eliminate stuttering
- Background worker thread for non-blocking torch.save() operations
- Queue-based system with maxsize=2 to prevent memory issues

### v1.2-professional-viz
- Web interface with Plotly Dash (real-time dashboard)
- Pygame high-performance renderer (60+ FPS)
- Chemical field heatmap visualization (4 overlays)
- Interactive controls and camera system
- 3 rendering options: matplotlib, web, pygame

### v1.1-chemical-signaling
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

### 选项 1：Matplotlib 渲染器（默认）

```bash
# 使用 matplotlib 运行（较慢，适合分析）
python main.py

# 大网格
python main.py --grid 30

# 验证模式：测试训练网络 vs 随机网络
python main.py --validate
```

### 选项 2：Web 界面（推荐）

```bash
# 启动 Web 界面，实时化学场可视化
python web_interface.py

# 在浏览器打开 http://localhost:8050
# 功能：
# - 实时种群图表
# - 4 种化学场热力图
# - 物种统计表
# - 暂停/恢复/重置控制
```

### 选项 3：Pygame 渲染器（最快）

```bash
# 高性能渲染器，60+ FPS
python pygame_renderer.py

# 键盘控制：
# - SPACE：暂停/恢复
# - R：重置模拟
# - S：手动保存最佳网络
# - C：切换化学场覆盖层
# - G：切换网格
# - +/-：调整速度（1x 到 10x）
# - 鼠标滚轮：缩放
# - 点击拖拽：平移相机
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

### 基因组可视化（方案 C：混合基因组）
- **12 维基因组**：每个生物具有独特的基因组向量，结合：
  - 神经指纹（8 维）：从神经网络权重提取的统计特征（每层的均值、标准差、绝对均值、最大值）
  - 化学亲和力（4 维）：分泌不同化学物质的可进化偏好
- **基因组到颜色映射**：物种颜色根据其平均基因组动态生成
  - 相似基因组在 HSV 空间产生相似颜色
  - 颜色随神经网络和化学偏好的进化而连续演变
  - 创建遗传相关物种的直观视觉聚类
- **软过渡**：基因组系统与离散物种 ID 共存
  - 物种 ID 仍用于游戏机制（向后兼容）
  - 基因组仅用于可视化（可切换开关）
  - 每 50 代更新以反映进化变化

### 进化机制
1. **有性繁殖与基因交叉**：
   - 生物寻找配偶（同物种邻居）
   - 后代从父母各继承 50% 神经元（穿插交叉）
   - 无配偶时回退到无性克隆
   - 继承后施加变异（10% 变异率）
2. **强化学习**：成功行为（捕猎、逃脱、繁殖）强化神经网络
3. **优势物种分裂**：占比 >75% 的物种自动分裂防止竞争排斥
4. **地理物种分裂**：当物种分离成不同空间集群时分裂（每 100 代检查）

### 适应度与持久化
- 适应度 = 存活代数 + 繁殖次数 × 10
- 每 50 代保存最优网络到 `best_brain.pt`
- **异步保存**：使用后台线程避免保存时卡顿
- 启动时 20% 的细胞继承保存的权重
