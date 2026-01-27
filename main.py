"""
Digital Primordial Soup - A GPU-accelerated Artificial Life Simulation

This simulation implements a neural network-driven ecosystem where multiple species
compete for survival through predation, reproduction, and evolution. Key features:
- Neuroevolution: Neural network weights are inherited and mutated during reproduction
- Reinforcement Learning: Weights are fine-tuned within each generation based on rewards
- Dynamic Speciation: Dominant species split into new species to maintain diversity
- GPU Acceleration: All computations run on GPU (CUDA/MPS) when available
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import colorsys
import time
import os
import argparse

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================
parser = argparse.ArgumentParser(description='Digital Primordial Soup - Neural Evolution Simulation')
parser.add_argument('--validate', '-v', action='store_true',
                    help='Validation mode: S0 loads trained weights, S1/S2 use random weights')
parser.add_argument('--grid', '-g', type=int, default=20,
                    help='Grid size (default: 20)')
ARGS = parser.parse_args()

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
# Grid and energy
GRID_SIZE = ARGS.grid
INITIAL_ENERGY = 30.0
MAX_ENERGY = 100.0
MOVE_COST = 0.2
CROWDING_THRESHOLD = 4
CROWDING_PENALTY = 0.5

# Species configuration
INITIAL_NUM_SPECIES = 3
MAX_SPECIES = 10
SPECIES_METABOLISM = 0.1
SPECIES_REPRO_THRESHOLD = 20
SPECIES_REPRO_COST = 8
SPECIES_OFFSPRING_ENERGY = 25
SPECIES_STARVATION = 100
SPECIES_HIDDEN_SIZE = 8
HIDDEN_SIZE_INCREMENT = 2  # Neurons added on speciation
MAX_HIDDEN_SIZE = 240       # Maximum hidden layer size

# Combat
ATTACK_BONUS = 1.2  # Energy multiplier when eating prey

# Evolution
MUTATION_RATE = 0.1
DOMINANCE_THRESHOLD = 0.75  # Species split when >75% of population
SPLIT_MUTATION_RATE = 0.3   # Higher mutation during speciation

# Reinforcement Learning
RL_LEARNING_RATE = 0.01
REWARD_EAT_PREY = 2.0
REWARD_SURVIVE_ATTACK = 1.0
REWARD_REPRODUCE = 1.5

# Neural Network Persistence
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FILE = os.path.join(SAVE_DIR, "best_brain.pt")
SAVE_INTERVAL = 50
ELITE_RATIO = 0.2

# Neural Network Architecture
INPUT_SIZE = 20
NUM_ACTIONS = 7
ACTION_STAY = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_EAT = 5
ACTION_REPRODUCE = 6

# =============================================================================
# SPECIES MANAGEMENT
# =============================================================================
SPECIES_CONFIG = []
SPECIES_CHILD_COUNT = {}


def generate_random_color(index: int, total: int) -> tuple:
    """Generate a visually distinct color using HSV color space."""
    hue = (index / max(total, 1)) % 1.0
    saturation = 0.7 + np.random.random() * 0.3
    value = 0.7 + np.random.random() * 0.3
    return colorsys.hsv_to_rgb(hue, saturation, value)


def create_species_config(sp_id: int, num_total: int, name: str = None) -> dict:
    """Create configuration for a single species."""
    prey_list = [j for j in range(num_total) if j != sp_id]
    return {
        'name': name or f'S{sp_id}',
        'color': generate_random_color(sp_id, max(num_total, MAX_SPECIES)),
        'prey': prey_list,
        'metabolism': SPECIES_METABOLISM,
        'repro_threshold': SPECIES_REPRO_THRESHOLD,
        'repro_cost': SPECIES_REPRO_COST,
        'offspring_energy': SPECIES_OFFSPRING_ENERGY,
        'starvation': SPECIES_STARVATION,
        'hidden_size': SPECIES_HIDDEN_SIZE,
        'parent': None,
        'extinct': False,
    }


def add_new_species(parent_id: int) -> int:
    """Create a new species by splitting from a parent species."""
    new_id = len(SPECIES_CONFIG)
    if new_id >= MAX_SPECIES:
        return None

    # Generate lineage-based name: S{parent}_{child_num}
    parent_name = SPECIES_CONFIG[parent_id]['name']
    child_num = SPECIES_CHILD_COUNT.get(parent_id, 0) + 1
    SPECIES_CHILD_COUNT[parent_id] = child_num
    new_name = f"{parent_name}_{child_num}"

    new_config = create_species_config(new_id, new_id + 1, name=new_name)
    new_config['parent'] = parent_id

    # New species gets more hidden neurons (evolution of complexity)
    parent_hidden = SPECIES_CONFIG[parent_id]['hidden_size']
    new_hidden = min(parent_hidden + HIDDEN_SIZE_INCREMENT, MAX_HIDDEN_SIZE)
    new_config['hidden_size'] = new_hidden

    SPECIES_CONFIG.append(new_config)

    # Update all existing species' prey lists
    for sp_id in range(new_id):
        SPECIES_CONFIG[sp_id]['prey'].append(new_id)

    # Rebuild GPU tensors
    global SPECIES_TENSORS
    SPECIES_TENSORS = build_species_tensors()

    return new_id


# Initialize species
for i in range(INITIAL_NUM_SPECIES):
    SPECIES_CONFIG.append(create_species_config(i, INITIAL_NUM_SPECIES, name=f'S{i}'))
    SPECIES_CHILD_COUNT[i] = 0


def build_species_tensors():
    """Build GPU tensors for species parameters (called when species change)."""
    num_sp = len(SPECIES_CONFIG)

    # Parameter lookup tensors [MAX_SPECIES]
    metabolism = torch.zeros(MAX_SPECIES, dtype=torch.float32, device=DEVICE)
    repro_threshold = torch.zeros(MAX_SPECIES, dtype=torch.float32, device=DEVICE)
    repro_cost = torch.zeros(MAX_SPECIES, dtype=torch.float32, device=DEVICE)
    offspring_energy = torch.zeros(MAX_SPECIES, dtype=torch.float32, device=DEVICE)
    starvation = torch.zeros(MAX_SPECIES, dtype=torch.float32, device=DEVICE)
    hidden_sizes = torch.zeros(MAX_SPECIES, dtype=torch.int32, device=DEVICE)

    for sp_id in range(num_sp):
        sp = SPECIES_CONFIG[sp_id]
        metabolism[sp_id] = sp['metabolism']
        repro_threshold[sp_id] = sp['repro_threshold']
        repro_cost[sp_id] = sp['repro_cost']
        offspring_energy[sp_id] = sp['offspring_energy']
        starvation[sp_id] = sp['starvation']
        hidden_sizes[sp_id] = sp['hidden_size']

    # Prey matrix [MAX_SPECIES, MAX_SPECIES] - prey_matrix[pred][prey] = 1 if pred can eat prey
    prey_matrix = torch.zeros((MAX_SPECIES, MAX_SPECIES), dtype=torch.bool, device=DEVICE)
    for sp_id in range(num_sp):
        for prey_id in SPECIES_CONFIG[sp_id]['prey']:
            if prey_id < MAX_SPECIES:
                prey_matrix[sp_id, prey_id] = True

    return {
        'metabolism': metabolism,
        'repro_threshold': repro_threshold,
        'repro_cost': repro_cost,
        'offspring_energy': offspring_energy,
        'starvation': starvation,
        'hidden_sizes': hidden_sizes,
        'prey_matrix': prey_matrix,
        'num_species': num_sp,
    }


# Global species tensors (rebuilt when species split)
SPECIES_TENSORS = build_species_tensors()


# =============================================================================
# SIMULATION ENGINE
# =============================================================================
class GPULifeGame:
    """GPU-accelerated artificial life simulation with neural network evolution."""

    def __init__(self, size: int = GRID_SIZE):
        self.size = size
        self.generation = 0

        # State tensors
        self.alive = torch.zeros((size, size), dtype=torch.bool, device=DEVICE)
        self.energy = torch.zeros((size, size), dtype=torch.float32, device=DEVICE)
        self.species = torch.zeros((size, size), dtype=torch.int32, device=DEVICE)
        self.hunger = torch.zeros((size, size), dtype=torch.int32, device=DEVICE)
        self.is_newborn = torch.zeros((size, size), dtype=torch.bool, device=DEVICE)

        # Reinforcement learning tensors
        self.reward = torch.zeros((size, size), dtype=torch.float32, device=DEVICE)
        self.last_action = torch.zeros((size, size), dtype=torch.int64, device=DEVICE)
        self.last_action_logprob = torch.zeros((size, size), dtype=torch.float32, device=DEVICE)
        self.last_inputs = torch.zeros((size, size, INPUT_SIZE), dtype=torch.float32, device=DEVICE)
        self.last_hidden = torch.zeros((size, size, MAX_HIDDEN_SIZE), dtype=torch.float32, device=DEVICE)

        # Fitness tracking
        self.lifetime = torch.zeros((size, size), dtype=torch.int32, device=DEVICE)
        self.repro_count = torch.zeros((size, size), dtype=torch.int32, device=DEVICE)
        self.best_fitness = 0
        self.best_w1 = None
        self.best_w2 = None
        self.best_hidden_size = SPECIES_HIDDEN_SIZE

        # Neural network weights (one per cell)
        self.w1 = torch.randn((size, size, INPUT_SIZE, MAX_HIDDEN_SIZE), device=DEVICE) * 0.5
        self.w2 = torch.randn((size, size, MAX_HIDDEN_SIZE, NUM_ACTIONS), device=DEVICE) * 0.5

        # Position indices for vectorized operations
        self.rows = torch.arange(size, device=DEVICE).view(-1, 1).expand(size, size)
        self.cols = torch.arange(size, device=DEVICE).view(1, -1).expand(size, size)

        # Pre-allocated kernels and tensors for CUDA optimization
        self.crowding_kernel = torch.ones((1, 1, 3, 3), device=DEVICE)
        self.crowding_kernel[0, 0, 1, 1] = 0
        self.neuron_idx = torch.arange(MAX_HIDDEN_SIZE, device=DEVICE).view(1, 1, -1)

        # Load saved weights if available
        self.saved_w1, self.saved_w2, self.saved_hidden_size = self._load_saved_weights()
        self._spawn_initial_cells()

        self.history = {'population': [], 'species': [[] for _ in range(len(SPECIES_CONFIG))]}

    # -------------------------------------------------------------------------
    # Weight Persistence
    # -------------------------------------------------------------------------
    def _load_saved_weights(self):
        """Load previously saved neural network weights and structure."""
        if os.path.exists(SAVE_FILE):
            try:
                checkpoint = torch.load(SAVE_FILE, map_location=DEVICE, weights_only=True)
                w1 = checkpoint['w1']
                w2 = checkpoint['w2']
                hidden_size = checkpoint.get('hidden_size', SPECIES_HIDDEN_SIZE)
                fitness = checkpoint.get('fitness', 0)
                gen = checkpoint.get('generation', 0)
                print(f"Loaded saved network: fitness={fitness}, hidden={hidden_size}, from gen {gen}")

                # Pad or truncate weights to match MAX_HIDDEN_SIZE
                saved_hidden = w1.shape[-1]
                if saved_hidden != MAX_HIDDEN_SIZE:
                    print(f"  Adapting network: {saved_hidden} -> {MAX_HIDDEN_SIZE} neurons")
                    # Pad w1: [INPUT_SIZE, saved_hidden] -> [INPUT_SIZE, MAX_HIDDEN_SIZE]
                    new_w1 = torch.zeros((INPUT_SIZE, MAX_HIDDEN_SIZE), device=DEVICE)
                    copy_size = min(saved_hidden, MAX_HIDDEN_SIZE)
                    new_w1[:, :copy_size] = w1[:, :copy_size].to(DEVICE)
                    w1 = new_w1

                    # Pad w2: [saved_hidden, NUM_ACTIONS] -> [MAX_HIDDEN_SIZE, NUM_ACTIONS]
                    new_w2 = torch.zeros((MAX_HIDDEN_SIZE, NUM_ACTIONS), device=DEVICE)
                    new_w2[:copy_size, :] = w2[:copy_size, :].to(DEVICE)
                    w2 = new_w2

                return w1.to(DEVICE), w2.to(DEVICE), hidden_size
            except Exception as e:
                print(f"Warning: Failed to load weights: {e}")
                return None, None, SPECIES_HIDDEN_SIZE
        else:
            print("No saved weights found, using random initialization")
            return None, None, SPECIES_HIDDEN_SIZE

    def _save_best_weights(self):
        """Save the current best neural network weights and structure."""
        if self.best_w1 is None:
            return
        checkpoint = {
            'w1': self.best_w1.cpu(),
            'w2': self.best_w2.cpu(),
            'hidden_size': self.best_hidden_size,
            'fitness': self.best_fitness,
            'generation': self.generation,
        }
        torch.save(checkpoint, SAVE_FILE)
        print(f"Saved best network: fitness={self.best_fitness}, hidden={self.best_hidden_size}, gen {self.generation}")

    def _update_best_network(self):
        """Find and track the best performing individual."""
        if not self.alive.any():
            return

        # Fitness = lifetime + reproduction_count * 10
        fitness = self.lifetime.float() + self.repro_count.float() * 10
        fitness = torch.where(self.alive, fitness, torch.zeros_like(fitness))

        max_fitness = fitness.max().item()
        if max_fitness > self.best_fitness:
            best_pos = (fitness == max_fitness).nonzero(as_tuple=False)[0]
            r, c = best_pos[0].item(), best_pos[1].item()
            self.best_fitness = max_fitness
            self.best_w1 = self.w1[r, c].clone()
            self.best_w2 = self.w2[r, c].clone()
            # Track the hidden size of the best individual's species
            sp_id = self.species[r, c].item()
            self.best_hidden_size = SPECIES_CONFIG[sp_id]['hidden_size']

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def _spawn_initial_cells(self):
        """Spawn initial population, optionally with saved weights."""
        num_cells = int(self.size * self.size * 0.12)
        positions = torch.randperm(self.size * self.size, device=DEVICE)[:num_cells]
        rows = positions // self.size
        cols = positions % self.size
        self.alive[rows, cols] = True
        self.energy[rows, cols] = INITIAL_ENERGY
        self.species[rows, cols] = torch.randint(0, len(SPECIES_CONFIG), (num_cells,), dtype=torch.int32, device=DEVICE)

        if self.saved_w1 is None or self.saved_w2 is None:
            return

        if ARGS.validate:
            # Validation mode: only S0 uses trained weights
            species_tensor = self.species[rows, cols]
            s0_mask = (species_tensor == 0)
            s0_count = s0_mask.sum().item()
            if s0_count > 0:
                s0_rows = rows[s0_mask]
                s0_cols = cols[s0_mask]
                self.w1[s0_rows, s0_cols] = self.saved_w1.clone() + torch.randn_like(self.saved_w1) * 0.05
                self.w2[s0_rows, s0_cols] = self.saved_w2.clone() + torch.randn_like(self.saved_w2) * 0.05
                print(f"Validation mode: {s0_count} S0 cells use trained weights")
                print(f"                 S1/S2 use random weights (control group)")
        else:
            # Normal mode: elite cells use trained weights
            num_elite = int(num_cells * ELITE_RATIO)
            if num_elite > 0:
                elite_indices = torch.randperm(num_cells, device=DEVICE)[:num_elite]
                elite_rows = rows[elite_indices]
                elite_cols = cols[elite_indices]
                self.w1[elite_rows, elite_cols] = self.saved_w1.clone() + torch.randn_like(self.saved_w1) * 0.05
                self.w2[elite_rows, elite_cols] = self.saved_w2.clone() + torch.randn_like(self.saved_w2) * 0.05
                print(f"{num_elite} cells inherited saved network weights")

    # -------------------------------------------------------------------------
    # Neural Network
    # -------------------------------------------------------------------------
    def _get_shifted(self, tensor, dr, dc):
        """Shift tensor with toroidal (wrap-around) boundary conditions."""
        return torch.roll(torch.roll(tensor, -dr, 0), -dc, 1)

    def _build_inputs(self):
        """Build neural network input features from environment state."""
        alive_f = self.alive.float()
        energy_norm = self.energy / MAX_ENERGY
        species_f = self.species.float()

        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        neighbor_energy = torch.stack([self._get_shifted(energy_norm * alive_f, dr, dc) for dr, dc in dirs], dim=-1)
        neighbor_alive = torch.stack([self._get_shifted(alive_f, dr, dc) for dr, dc in dirs], dim=-1)
        neighbor_species = torch.stack([self._get_shifted(species_f * alive_f, dr, dc) for dr, dc in dirs], dim=-1)

        same = (neighbor_species == species_f.unsqueeze(-1)) * neighbor_alive
        same_count = same.sum(dim=-1, keepdim=True)
        diff_count = neighbor_alive.sum(dim=-1, keepdim=True) - same_count

        inputs = torch.cat([
            neighbor_energy,                                    # 8: neighbor energy levels
            same_count / 8.0,                                   # 1: same species count
            diff_count / 8.0,                                   # 1: different species count
            energy_norm.unsqueeze(-1),                          # 1: own energy
            (neighbor_alive.sum(dim=-1) / 8.0).unsqueeze(-1),   # 1: total neighbor count
            torch.zeros((self.size, self.size, 8), device=DEVICE)  # 8: padding
        ], dim=-1)
        return inputs

    def _batch_forward(self, inputs):
        """Batch forward pass through all cells' neural networks (CUDA optimized)."""
        h = torch.tanh(torch.einsum('ijk,ijkl->ijl', inputs, self.w1))

        # Apply species-specific hidden layer mask (vectorized lookup)
        hidden_sizes = self._get_species_param_fast(SPECIES_TENSORS['hidden_sizes'].float()).int()
        mask = (self.neuron_idx < hidden_sizes.unsqueeze(-1)).float()
        h = h * mask

        logits = torch.einsum('ijk,ijkl->ijl', h, self.w2)
        probs = F.softmax(logits, dim=-1)

        self.last_inputs = inputs
        self.last_hidden = h
        return probs

    def _sample_actions(self, probs):
        """Sample actions from probability distributions."""
        flat_probs = probs.view(-1, NUM_ACTIONS)
        actions = torch.multinomial(flat_probs, 1).view(self.size, self.size)

        log_probs = torch.log(probs + 1e-8)
        action_logprobs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        self.last_action = actions
        self.last_action_logprob = action_logprobs
        return actions

    # -------------------------------------------------------------------------
    # Game Logic (CUDA Optimized)
    # -------------------------------------------------------------------------
    def _get_species_param_fast(self, param_tensor):
        """Get species-specific parameter using pre-built lookup tensor (O(1) per cell)."""
        # param_tensor is [MAX_SPECIES], self.species is [H, W]
        # Use advanced indexing for vectorized lookup
        species_clamped = self.species.clamp(0, MAX_SPECIES - 1).long()
        return param_tensor[species_clamped]

    def _is_valid_prey_fast(self, predator_species, neighbor_species):
        """Check if neighbors are valid prey using pre-built prey matrix (O(1))."""
        # prey_matrix[pred, prey] = True if pred can eat prey
        pred_clamped = predator_species.clamp(0, MAX_SPECIES - 1).long()
        prey_clamped = neighbor_species.clamp(0, MAX_SPECIES - 1).long()
        # Flatten for 2D indexing, then reshape
        flat_pred = pred_clamped.view(-1)
        flat_prey = prey_clamped.view(-1)
        result = SPECIES_TENSORS['prey_matrix'][flat_pred, flat_prey]
        return result.view(predator_species.shape)

    def step(self):
        """Execute one simulation step (CUDA optimized)."""
        self.is_newborn.fill_(False)

        # Metabolism (vectorized lookup)
        metabolism_map = self._get_species_param_fast(SPECIES_TENSORS['metabolism'])
        self.energy = torch.where(self.alive, self.energy - metabolism_map, self.energy)

        # Crowding penalty (already optimized with conv2d)
        alive_f = self.alive.float()
        padded = F.pad(alive_f.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='circular')
        neighbors = F.conv2d(padded, self.crowding_kernel).squeeze()
        crowding_cost = torch.clamp(neighbors - CROWDING_THRESHOLD, min=0) * CROWDING_PENALTY
        self.energy = torch.where(self.alive, self.energy - crowding_cost, self.energy)

        # Starvation (vectorized lookup)
        self.hunger = torch.where(self.alive, self.hunger + 1, self.hunger)
        starvation_map = self._get_species_param_fast(SPECIES_TENSORS['starvation'])
        self.energy = torch.where(self.hunger >= starvation_map.int(), torch.zeros_like(self.energy), self.energy)

        # Neural network decision
        inputs = self._build_inputs()
        probs = self._batch_forward(inputs)
        actions = self._sample_actions(probs)

        # Execute actions
        self._parallel_actions(actions)

        # Update lifetime
        self.lifetime = torch.where(self.alive, self.lifetime + 1, self.lifetime)

        # Reset stats on death
        dying = self.alive & (self.energy <= 0)
        self.lifetime[dying] = 0
        self.repro_count[dying] = 0

        # Death
        self.alive = self.alive & (self.energy > 0)

        # RL update
        self._apply_rl_update()

        # Species splitting
        self._check_and_split_dominant_species()

        # Save best network periodically
        self._update_best_network()
        if self.generation > 0 and self.generation % SAVE_INTERVAL == 0:
            self._save_best_weights()

        self.generation += 1

        # Statistics (vectorized with bincount)
        alive_species = self.species[self.alive].long()
        if len(alive_species) > 0:
            counts = torch.bincount(alive_species, minlength=MAX_SPECIES)
            total = len(alive_species)
        else:
            counts = torch.zeros(MAX_SPECIES, dtype=torch.long, device=DEVICE)
            total = 0

        for sp_id in range(len(SPECIES_CONFIG)):
            self.history['species'][sp_id].append(counts[sp_id].item())
        self.history['population'].append(total)

    def _parallel_actions(self, actions):
        """Execute all actions in parallel."""
        # Movement
        for action, (dr, dc) in [(ACTION_UP, (-1, 0)), (ACTION_DOWN, (1, 0)),
                                  (ACTION_LEFT, (0, -1)), (ACTION_RIGHT, (0, 1))]:
            is_this_move = self.alive & (actions == action)
            if not is_this_move.any():
                continue

            self.energy = torch.where(is_this_move, self.energy - MOVE_COST, self.energy)
            target_r = (self.rows + dr) % self.size
            target_c = (self.cols + dc) % self.size
            can_move = is_this_move & ~self.alive[target_r, target_c]
            random_pass = torch.rand((self.size, self.size), device=DEVICE) > 0.5
            winner = can_move & random_pass

            if winner.any():
                move_energy = self.energy[winner]
                move_species = self.species[winner]
                move_hunger = self.hunger[winner]
                move_w1 = self.w1[winner]
                move_w2 = self.w2[winner]

                self.alive[winner] = False
                self.energy[winner] = 0

                new_r = target_r[winner]
                new_c = target_c[winner]
                self.alive[new_r, new_c] = True
                self.energy[new_r, new_c] = move_energy
                self.species[new_r, new_c] = move_species
                self.hunger[new_r, new_c] = move_hunger
                self.w1[new_r, new_c] = move_w1
                self.w2[new_r, new_c] = move_w2

        # Eating
        is_eat = self.alive & (actions == ACTION_EAT)
        if is_eat.any():
            self._parallel_eat(is_eat)

        # Reproduction
        repro_threshold = self._get_species_param_fast(SPECIES_TENSORS['repro_threshold'])
        is_reproduce = self.alive & (actions == ACTION_REPRODUCE) & (self.energy >= repro_threshold)
        if is_reproduce.any():
            self._parallel_reproduce(is_reproduce)

    def _parallel_eat(self, is_eat):
        """Handle predation between species."""
        my_species = self.species

        dirs_basic = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dirs_extended = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]

        # Build predator mask
        has_prey_mask = torch.zeros_like(my_species, dtype=torch.bool)
        for sp_id in range(len(SPECIES_CONFIG)):
            if SPECIES_CONFIG[sp_id]['prey']:
                has_prey_mask = has_prey_mask | (my_species == sp_id)
        predator_eat = is_eat & has_prey_mask

        # Process basic directions first
        for dr, dc in dirs_basic:
            neighbor_r = (self.rows + dr) % self.size
            neighbor_c = (self.cols + dc) % self.size
            neighbor_alive = self.alive[neighbor_r, neighbor_c]
            neighbor_species = self.species[neighbor_r, neighbor_c]
            neighbor_energy = self.energy[neighbor_r, neighbor_c]

            is_valid_prey = self._is_valid_prey_fast(my_species, neighbor_species)
            can_attack = predator_eat & neighbor_alive & is_valid_prey
            attack_success = torch.rand_like(self.energy) < 0.5
            can_eat = can_attack & attack_success

            # Reward for escaping
            escaped = can_attack & ~attack_success
            if escaped.any():
                escape_prey_r = neighbor_r[escaped]
                escape_prey_c = neighbor_c[escaped]
                self.reward[escape_prey_r, escape_prey_c] += REWARD_SURVIVE_ATTACK

            if can_eat.any():
                gained = neighbor_energy[can_eat] * ATTACK_BONUS
                self.energy[can_eat] = torch.clamp(self.energy[can_eat] + gained, max=MAX_ENERGY)
                self.hunger[can_eat] = 0
                self.reward[can_eat] += REWARD_EAT_PREY
                prey_r = neighbor_r[can_eat]
                prey_c = neighbor_c[can_eat]
                self.alive[prey_r, prey_c] = False
                self.energy[prey_r, prey_c] = 0
                predator_eat = predator_eat & ~can_eat

        # Extended directions (diagonal and long-range)
        for dr, dc in dirs_extended:
            neighbor_r = (self.rows + dr) % self.size
            neighbor_c = (self.cols + dc) % self.size
            neighbor_alive = self.alive[neighbor_r, neighbor_c]
            neighbor_species = self.species[neighbor_r, neighbor_c]
            neighbor_energy = self.energy[neighbor_r, neighbor_c]

            is_valid_prey = self._is_valid_prey_fast(my_species, neighbor_species)
            can_attack = predator_eat & neighbor_alive & is_valid_prey
            attack_success = torch.rand_like(self.energy) < 0.5
            can_eat = can_attack & attack_success

            escaped = can_attack & ~attack_success
            if escaped.any():
                escape_prey_r = neighbor_r[escaped]
                escape_prey_c = neighbor_c[escaped]
                self.reward[escape_prey_r, escape_prey_c] += REWARD_SURVIVE_ATTACK

            if can_eat.any():
                gained = neighbor_energy[can_eat] * ATTACK_BONUS
                self.energy[can_eat] = torch.clamp(self.energy[can_eat] + gained, max=MAX_ENERGY)
                self.hunger[can_eat] = 0
                self.reward[can_eat] += REWARD_EAT_PREY
                prey_r = neighbor_r[can_eat]
                prey_c = neighbor_c[can_eat]
                self.alive[prey_r, prey_c] = False
                self.energy[prey_r, prey_c] = 0
                predator_eat = predator_eat & ~can_eat

    def _parallel_reproduce(self, is_reproduce):
        """Handle reproduction with inheritance and mutation (CUDA optimized)."""
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

        # Vectorized parameter lookups
        repro_threshold = self._get_species_param_fast(SPECIES_TENSORS['repro_threshold'])
        repro_cost = self._get_species_param_fast(SPECIES_TENSORS['repro_cost'])
        offspring_energy = self._get_species_param_fast(SPECIES_TENSORS['offspring_energy'])

        for dr, dc in dirs:
            target_r = (self.rows + dr) % self.size
            target_c = (self.cols + dc) % self.size
            target_empty = ~self.alive[target_r, target_c]
            can_reproduce = is_reproduce & target_empty & (self.energy >= repro_threshold)

            if can_reproduce.any():
                random_pass = torch.rand((self.size, self.size), device=DEVICE) > 0.5
                winner = can_reproduce & random_pass

                if winner.any():
                    self.energy[winner] -= repro_cost[winner]
                    self.reward[winner] += REWARD_REPRODUCE
                    self.repro_count[winner] += 1

                    child_r = target_r[winner]
                    child_c = target_c[winner]
                    self.alive[child_r, child_c] = True
                    self.energy[child_r, child_c] = offspring_energy[winner]
                    self.species[child_r, child_c] = self.species[winner]
                    self.hunger[child_r, child_c] = 0
                    self.is_newborn[child_r, child_c] = True

                    # Inheritance with mutation
                    self.w1[child_r, child_c] = self.w1[winner] + torch.randn_like(self.w1[winner]) * MUTATION_RATE
                    self.w2[child_r, child_c] = self.w2[winner] + torch.randn_like(self.w2[winner]) * MUTATION_RATE

                    self.reward[child_r, child_c] = 0
                    self.lifetime[child_r, child_c] = 0
                    self.repro_count[child_r, child_c] = 0
                    is_reproduce = is_reproduce & ~winner

    def _apply_rl_update(self):
        """Apply policy gradient updates based on rewards (CUDA optimized - fully vectorized)."""
        has_reward = self.alive & (self.reward != 0)
        if not has_reward.any():
            return

        # Get cells with rewards
        reward_mask = has_reward
        rewards = self.reward[reward_mask]  # [N]
        actions = self.last_action[reward_mask]  # [N]
        hidden = self.last_hidden[reward_mask]  # [N, H]
        inputs = self.last_inputs[reward_mask]  # [N, I]

        # Limit to avoid memory issues
        n = min(len(rewards), 2000)
        if n < len(rewards):
            perm = torch.randperm(len(rewards), device=DEVICE)[:n]
            rewards = rewards[perm]
            actions = actions[perm]
            hidden = hidden[perm]
            inputs = inputs[perm]
            # Get corresponding positions
            positions = reward_mask.nonzero(as_tuple=False)[perm]
        else:
            positions = reward_mask.nonzero(as_tuple=False)

        rows, cols = positions[:, 0], positions[:, 1]

        # Vectorized w2 update: w2[r, c, :, action] += lr * reward * h
        # Create one-hot action vectors [N, NUM_ACTIONS]
        action_onehot = F.one_hot(actions, NUM_ACTIONS).float()  # [N, A]
        # delta_w2[n, h, a] = lr * reward[n] * hidden[n, h] * onehot[n, a]
        delta_w2 = RL_LEARNING_RATE * rewards.view(-1, 1, 1) * hidden.unsqueeze(-1) * action_onehot.unsqueeze(1)
        # Apply updates
        self.w2[rows, cols] += delta_w2

        # Vectorized w1 update (simplified: update all neurons proportional to activation)
        # delta_w1[n, i, h] = lr * 0.1 * reward[n] * input[n, i] * sign(h[n, h]) * (|h| > 0.1)
        h_sign = torch.sign(hidden)  # [N, H]
        h_active = (hidden.abs() > 0.1).float()  # [N, H]
        # delta_w1[n, i, h] = lr * 0.1 * reward[n] * input[n, i] * sign[n, h] * active[n, h]
        delta_w1 = RL_LEARNING_RATE * 0.1 * rewards.view(-1, 1, 1) * inputs.unsqueeze(-1) * (h_sign * h_active).unsqueeze(1)
        self.w1[rows, cols] += delta_w1

        self.reward.fill_(0)

    def _check_and_split_dominant_species(self):
        """Split dominant species to maintain ecosystem diversity (CUDA optimized)."""
        total_alive = self.alive.sum().item()
        if total_alive < 10:
            return False

        # Vectorized species counting with bincount
        alive_species = self.species[self.alive].long()
        counts = torch.bincount(alive_species, minlength=MAX_SPECIES)
        species_counts = counts[:len(SPECIES_CONFIG)].tolist()

        max_count = max(species_counts)
        dominant_species = species_counts.index(max_count)
        dominance_ratio = max_count / total_alive

        if dominance_ratio < DOMINANCE_THRESHOLD:
            return False

        # Mark extinct species
        for sp_id, count in enumerate(species_counts):
            if count == 0 and not SPECIES_CONFIG[sp_id]['extinct']:
                SPECIES_CONFIG[sp_id]['extinct'] = True

        new_species_id = add_new_species(dominant_species)
        if new_species_id is None:
            print(f"\nMax species limit ({MAX_SPECIES}) reached, cannot split")
            return False

        parent_name = SPECIES_CONFIG[dominant_species]['name']
        new_name = SPECIES_CONFIG[new_species_id]['name']
        new_hidden = SPECIES_CONFIG[new_species_id]['hidden_size']
        parent_hidden = SPECIES_CONFIG[dominant_species]['hidden_size']
        print(f"\n[SPECIATION] {parent_name} at {dominance_ratio*100:.1f}% > {DOMINANCE_THRESHOLD*100}%")
        print(f"             {parent_name} splits into {new_name}")
        print(f"             Hidden neurons: {parent_hidden} -> {new_hidden}")
        print(f"             Total species: {len(SPECIES_CONFIG)}")

        self.history['species'].append([0] * len(self.history['population']))

        dominant_mask = (self.species == dominant_species) & self.alive
        dominant_positions = dominant_mask.nonzero(as_tuple=False)

        num_to_split = len(dominant_positions) // 2
        if num_to_split == 0:
            return False

        perm = torch.randperm(len(dominant_positions), device=DEVICE)
        split_indices = dominant_positions[perm[:num_to_split]]

        split_r = split_indices[:, 0]
        split_c = split_indices[:, 1]
        self.species[split_r, split_c] = new_species_id

        # Apply higher mutation during speciation
        mutation_w1 = torch.randn_like(self.w1[split_r, split_c]) * SPLIT_MUTATION_RATE
        mutation_w2 = torch.randn_like(self.w2[split_r, split_c]) * SPLIT_MUTATION_RATE
        self.w1[split_r, split_c] += mutation_w1
        self.w2[split_r, split_c] += mutation_w2

        print(f"             {num_to_split} cells become {new_name}")
        return True

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------
    def render(self):
        """Render simulation state as RGB image."""
        img = np.zeros((self.size, self.size, 3), dtype=np.float32)

        alive = self.alive.cpu().numpy()
        energy = self.energy.cpu().numpy()
        species = self.species.cpu().numpy()
        newborn = self.is_newborn.cpu().numpy()

        energy_norm = np.clip(energy / MAX_ENERGY, 0, 1)

        for sp_id in range(len(SPECIES_CONFIG)):
            sp_mask = alive & (species == sp_id)
            if not sp_mask.any():
                continue
            base_color = SPECIES_CONFIG[sp_id]['color']
            for c in range(3):
                img[sp_mask, c] = base_color[c] * (0.3 + 0.7 * energy_norm[sp_mask])

        newborn_mask = alive & newborn
        img[newborn_mask] = np.clip(img[newborn_mask] + 0.4, 0, 1)

        return img

    def is_extinct(self):
        """Check if all life has died."""
        return self.alive.sum().item() == 0


# =============================================================================
# VISUALIZATION
# =============================================================================
def print_system_info():
    """Print simulation configuration."""
    print("=" * 60)
    if ARGS.validate:
        print("Digital Primordial Soup - VALIDATION MODE")
        print("=" * 60)
        print("\n[Validation Mode]")
        print("  S0: Uses trained network weights (experimental group)")
        print("  S1, S2: Use random weights (control group)")
        print("  Observe if S0 outcompetes random species")
    else:
        print("Digital Primordial Soup - Neural Evolution Simulation")
        print("=" * 60)
    print(f"\n[{len(SPECIES_CONFIG)} Species - Mutual Predation]")
    print("  All species have identical parameters")
    print("  Differentiation comes from neural network weights")
    print("  50% attack success rate")
    print(f"\n[Shared Parameters]")
    print(f"  Metabolism: {SPECIES_METABOLISM}")
    print(f"  Reproduction threshold: {SPECIES_REPRO_THRESHOLD}")
    print(f"  Reproduction cost: {SPECIES_REPRO_COST}")
    print(f"  Offspring energy: {SPECIES_OFFSPRING_ENERGY}")
    print(f"  Starvation threshold: {SPECIES_STARVATION}")
    print(f"  Hidden layer size: {SPECIES_HIDDEN_SIZE}")
    print(f"\n[Hybrid Evolution + RL]")
    print(f"  Evolution: Inherit weights on reproduction (mutation rate {MUTATION_RATE})")
    print(f"  RL: Fine-tune weights within generation (learning rate {RL_LEARNING_RATE})")
    print(f"  Rewards:")
    print(f"    Successful hunt: +{REWARD_EAT_PREY}")
    print(f"    Escape attack: +{REWARD_SURVIVE_ATTACK}")
    print(f"    Reproduction: +{REWARD_REPRODUCE}")
    print(f"\n[Dynamic Speciation]")
    print(f"  Splits when single species exceeds {DOMINANCE_THRESHOLD*100:.0f}%")
    print(f"  Split mutation rate: {SPLIT_MUTATION_RATE}")
    print(f"  Initial species: {INITIAL_NUM_SPECIES}, Max species: {MAX_SPECIES}")
    print(f"  Hidden neurons: {SPECIES_HIDDEN_SIZE} -> +{HIDDEN_SIZE_INCREMENT}/split (max {MAX_HIDDEN_SIZE})")
    print(f"\n[Network Persistence]")
    print(f"  Save best network every {SAVE_INTERVAL} generations to: {SAVE_FILE}")
    print(f"  Fitness = lifetime + reproduction_count * 10")
    print(f"  {ELITE_RATIO*100:.0f}% of initial cells inherit saved weights")
    print("=" * 60 + "\n")


def main():
    """Main entry point: create simulation and display animation."""
    print_system_info()
    game = GPULifeGame()

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], width_ratios=[2, 1], hspace=0.15, wspace=0.1)

    # Top: Global statistics
    ax_global = fig.add_subplot(gs[0, :])
    global_species_lines = []
    for sp_id in range(len(SPECIES_CONFIG)):
        color = SPECIES_CONFIG[sp_id]['color']
        name = SPECIES_CONFIG[sp_id]['name']
        line, = ax_global.plot([], [], color=color, label=name, linewidth=2)
        global_species_lines.append(line)
    line_global_total, = ax_global.plot([], [], 'k--', label='Total', linewidth=1.5)
    ax_global.set_xlim(0, 100)
    ax_global.set_ylim(0, game.size * game.size * 0.15)
    ax_global.set_xlabel('Generation')
    ax_global.set_ylabel('Population', color='black')
    ax_global.grid(True, alpha=0.3)
    lines = global_species_lines + [line_global_total]
    labels = [l.get_label() for l in lines]
    ax_global.legend(lines, labels, loc='upper right', fontsize=8, ncol=5)
    ax_global.set_title('Global Ecosystem Dynamics')

    # Bottom left: Main display
    ax_main = fig.add_subplot(gs[1, 0])
    img_display = ax_main.imshow(game.render(), interpolation='nearest')
    mode_str = "VALIDATE" if ARGS.validate else ""
    ax_main.set_title(f'Digital Primordial Soup {mode_str} - Gen 0')
    ax_main.axis('off')

    # Bottom right: Species statistics
    ax_stats = fig.add_subplot(gs[1, 1])
    species_lines = []
    for sp_id in range(len(SPECIES_CONFIG)):
        color = SPECIES_CONFIG[sp_id]['color']
        name = SPECIES_CONFIG[sp_id]['name']
        line, = ax_stats.plot([], [], color=color, label=f'{sp_id}: {name}', linewidth=1.5)
        species_lines.append(line)
    line_total, = ax_stats.plot([], [], 'k--', label='Total', linewidth=1)
    ax_stats.set_xlim(0, 100)
    ax_stats.set_ylim(0, game.size * game.size * 0.15)
    ax_stats.set_xlabel('Generation')
    ax_stats.set_ylabel('Population')
    ax_stats.legend(loc='upper right', fontsize=7)
    ax_stats.set_title('Species Dynamics')
    ax_stats.grid(True, alpha=0.3)

    # Info text overlay
    info_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes,
                              fontsize=8, verticalalignment='top',
                              color='white', family='monospace',
                              bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    def update(_frame):
        nonlocal species_lines, global_species_lines

        start = time.time()
        game.step()
        elapsed = time.time() - start

        # Add new species lines if needed
        while len(species_lines) < len(SPECIES_CONFIG):
            sp_id = len(species_lines)
            color = SPECIES_CONFIG[sp_id]['color']
            name = SPECIES_CONFIG[sp_id]['name']
            line, = ax_stats.plot([], [], color=color, label=f'{sp_id}: {name}', linewidth=1.5)
            species_lines.append(line)
            line2, = ax_global.plot([], [], color=color, label=name, linewidth=2)
            global_species_lines.append(line2)
            ax_stats.legend(loc='upper right', fontsize=7)
            ax_global.legend(loc='upper right', fontsize=8, ncol=min(len(SPECIES_CONFIG), 5))

        # Update display
        img_display.set_array(game.render())
        ax_main.set_title(f'Digital Primordial Soup {mode_str} - Gen {game.generation} ({len(SPECIES_CONFIG)} species)')

        # Update charts
        gens = list(range(len(game.history['population'])))
        for sp_id in range(len(species_lines)):
            if sp_id < len(game.history['species']):
                species_lines[sp_id].set_data(gens, game.history['species'][sp_id])
        line_total.set_data(gens, game.history['population'])

        if len(gens) > 100:
            ax_stats.set_xlim(len(gens) - 100, len(gens))
        ax_global.set_xlim(0, max(len(gens), 100))

        if game.history['population']:
            max_pop = max(game.history['population']) * 1.2
            ax_stats.set_ylim(0, max(max_pop, 100))
            ax_global.set_ylim(0, max(max_pop, 100))

        for sp_id in range(len(global_species_lines)):
            if sp_id < len(game.history['species']):
                global_species_lines[sp_id].set_data(gens, game.history['species'][sp_id])
        line_global_total.set_data(gens, game.history['population'])

        # Update legend labels for extinct species
        legend_needs_update = False
        for sp_id in range(len(SPECIES_CONFIG)):
            is_extinct = SPECIES_CONFIG[sp_id].get('extinct', False)
            name = SPECIES_CONFIG[sp_id]['name']

            if sp_id < len(species_lines):
                current_label = species_lines[sp_id].get_label()
                expected_label = f"✗{sp_id}: {name}" if is_extinct else f"{sp_id}: {name}"
                if current_label != expected_label:
                    species_lines[sp_id].set_label(expected_label)
                    legend_needs_update = True

            if sp_id < len(global_species_lines):
                current_label = global_species_lines[sp_id].get_label()
                expected_label = f"✗{name}" if is_extinct else name
                if current_label != expected_label:
                    global_species_lines[sp_id].set_label(expected_label)
                    legend_needs_update = True

        if legend_needs_update:
            ax_stats.legend(loc='upper right', fontsize=7)
            ax_global.legend(loc='upper right', fontsize=8, ncol=min(len(SPECIES_CONFIG), 5))

        # Update info text
        total = game.history['population'][-1] if game.history['population'] else 0
        alive_species = sum(1 for sp in SPECIES_CONFIG if not sp.get('extinct', False))
        info = f"Gen: {game.generation} | Alive: {alive_species}/{len(SPECIES_CONFIG)} | Pop: {total}\n"
        for sp_id in range(len(SPECIES_CONFIG)):
            if sp_id < len(game.history['species']) and game.history['species'][sp_id]:
                count = game.history['species'][sp_id][-1]
            else:
                count = 0
            name = SPECIES_CONFIG[sp_id]['name']
            if count == 0:
                SPECIES_CONFIG[sp_id]['extinct'] = True
                info += f"✗ {name}\n"
            else:
                info += f"{name}: {count}\n"
        info += f"Step: {elapsed*1000:.1f}ms"
        info_text.set_text(info)

        if game.is_extinct():
            info_text.set_text(info + "\n\nEXTINCT!")

        return [img_display, *species_lines, line_total, *global_species_lines, line_global_total, info_text]

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
