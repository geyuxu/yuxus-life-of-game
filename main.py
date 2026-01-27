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
from scipy import ndimage

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
GRID_SIZE = 500 #ARGS.grid
INITIAL_ENERGY = 30.0
MAX_ENERGY = 100.0
MOVE_COST = 0.2
CROWDING_THRESHOLD = 4
CROWDING_PENALTY = 0.5

# Species configuration
INITIAL_NUM_SPECIES = 10
MAX_SPECIES = 50
SPECIES_METABOLISM = 0.1
SPECIES_REPRO_THRESHOLD = 20
SPECIES_REPRO_COST = 8
SPECIES_OFFSPRING_ENERGY = 25
SPECIES_STARVATION = 100
SPECIES_HIDDEN_SIZE = 8
HIDDEN_SIZE_INCREMENT = 2  # Neurons added on speciation
MAX_HIDDEN_SIZE = 240       # Maximum hidden layer size
MIN_HIDDEN_SIZE = 4         # Minimum hidden layer size

# Random mutation
MUTATION_INTERVAL = 10      # Check for random mutation every N generations
RANDOM_MUTATION_CHANCE = 0.1  # Probability per species per check

# Species recycling
EXTINCT_RECYCLE_DELAY = 50  # Generations before extinct species slot can be reused

# Blob-based speciation
BLOB_SEPARATION_DELAY = 50  # Generations of separation before blob becomes new species
BLOB_CHECK_INTERVAL = 25    # Check for blob separation every N generations (higher = less lag)

# Performance tuning
HISTORY_WINDOW = 1000       # Keep only last N generations in history (0 = unlimited)
RENDER_INTERVAL = 1         # Render every N frames (increase for faster simulation)
CHART_UPDATE_INTERVAL = 5   # Update charts every N frames (reduces matplotlib overhead)

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
        'extinct_gen': None,  # Generation when species went extinct (for recycling)
    }


def find_recyclable_species(current_gen: int) -> int:
    """
    Find a species slot that can be recycled (extinct for EXTINCT_RECYCLE_DELAY generations).

    Returns:
        The ID of a recyclable species, or None if none available
    """
    for sp_id, sp in enumerate(SPECIES_CONFIG):
        if sp['extinct'] and sp['extinct_gen'] is not None:
            if current_gen - sp['extinct_gen'] >= EXTINCT_RECYCLE_DELAY:
                return sp_id
    return None


def create_child_species(parent_id: int, hidden_delta: int = HIDDEN_SIZE_INCREMENT, current_gen: int = 0) -> int:
    """
    Create a new child species from a parent species.

    Args:
        parent_id: The ID of the parent species
        hidden_delta: Change in hidden layer size (can be positive or negative)
        current_gen: Current generation (for species recycling)

    Returns:
        The ID of the new species, or None if max species reached
    """
    # First, try to recycle an extinct species slot
    recycled_id = find_recyclable_species(current_gen)

    if recycled_id is not None:
        # Recycle the extinct species slot
        new_id = recycled_id
        old_name = SPECIES_CONFIG[new_id]['name']

        # Generate lineage-based name
        parent_name = SPECIES_CONFIG[parent_id]['name']
        child_num = SPECIES_CHILD_COUNT.get(parent_id, 0) + 1
        SPECIES_CHILD_COUNT[parent_id] = child_num
        new_name = f"{parent_name}_{child_num}"

        # Reset the species config
        parent_hidden = SPECIES_CONFIG[parent_id]['hidden_size']
        new_hidden = max(MIN_HIDDEN_SIZE, min(parent_hidden + hidden_delta, MAX_HIDDEN_SIZE))

        SPECIES_CONFIG[new_id] = {
            'name': new_name,
            'color': generate_random_color(new_id, MAX_SPECIES),
            'prey': [j for j in range(len(SPECIES_CONFIG)) if j != new_id],
            'metabolism': SPECIES_METABOLISM,
            'repro_threshold': SPECIES_REPRO_THRESHOLD,
            'repro_cost': SPECIES_REPRO_COST,
            'offspring_energy': SPECIES_OFFSPRING_ENERGY,
            'starvation': SPECIES_STARVATION,
            'hidden_size': new_hidden,
            'parent': parent_id,
            'extinct': False,
            'extinct_gen': None,
        }

        print(f"             [Recycled slot {new_id} from extinct {old_name}]")

    else:
        # Create new species slot
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

        # Apply hidden layer delta (can gain or lose neurons)
        parent_hidden = SPECIES_CONFIG[parent_id]['hidden_size']
        new_hidden = max(MIN_HIDDEN_SIZE, min(parent_hidden + hidden_delta, MAX_HIDDEN_SIZE))
        new_config['hidden_size'] = new_hidden

        SPECIES_CONFIG.append(new_config)

        # Update all existing species' prey lists
        for sp_id in range(new_id):
            SPECIES_CONFIG[sp_id]['prey'].append(new_id)

    # Rebuild GPU tensors
    global SPECIES_TENSORS
    SPECIES_TENSORS = build_species_tensors()

    return new_id


def add_new_species(parent_id: int, current_gen: int = 0) -> int:
    """Create a new species by splitting (always gains neurons). Kept for backward compatibility."""
    return create_child_species(parent_id, hidden_delta=HIDDEN_SIZE_INCREMENT, current_gen=current_gen)


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

        # Blob separation tracking: {species_id: {blob_hash: first_seen_gen}}
        # When a species has multiple blobs, track when each blob was first detected
        self.blob_separation_tracker = {}

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

        # Random mutation (every MUTATION_INTERVAL generations)
        self._check_random_mutation()

        # Blob separation speciation (geographic isolation)
        self._check_blob_separation()

        # Species splitting (dominant species)
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

        # Sliding window: trim old history to reduce memory
        if HISTORY_WINDOW > 0 and len(self.history['population']) > HISTORY_WINDOW:
            trim = len(self.history['population']) - HISTORY_WINDOW
            self.history['population'] = self.history['population'][trim:]
            for sp_id in range(len(self.history['species'])):
                if len(self.history['species'][sp_id]) > HISTORY_WINDOW:
                    self.history['species'][sp_id] = self.history['species'][sp_id][trim:]

    def _parallel_actions(self, actions):
        """Execute all actions in parallel (optimized: reduced .any() calls, shared random tensor)."""
        # Pre-generate shared random tensor for movement/reproduction
        rand_shared = torch.rand((self.size, self.size), device=DEVICE)

        # Movement (unified loop, removed .any() checks)
        for action, (dr, dc) in [(ACTION_UP, (-1, 0)), (ACTION_DOWN, (1, 0)),
                                  (ACTION_LEFT, (0, -1)), (ACTION_RIGHT, (0, 1))]:
            is_this_move = self.alive & (actions == action)

            # Apply move cost (no-op if mask is empty)
            self.energy = torch.where(is_this_move, self.energy - MOVE_COST, self.energy)
            target_r = (self.rows + dr) % self.size
            target_c = (self.cols + dc) % self.size
            can_move = is_this_move & ~self.alive[target_r, target_c]
            winner = can_move & (rand_shared > 0.5)

            # Direct tensor indexing (empty mask = no-op, no .any() needed)
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

        # Eating (always call, method handles empty mask efficiently)
        is_eat = self.alive & (actions == ACTION_EAT)
        self._parallel_eat(is_eat)

        # Reproduction (always call, method handles empty mask efficiently)
        repro_threshold = self._get_species_param_fast(SPECIES_TENSORS['repro_threshold'])
        is_reproduce = self.alive & (actions == ACTION_REPRODUCE) & (self.energy >= repro_threshold)
        self._parallel_reproduce(is_reproduce, rand_shared)

    def _parallel_eat(self, is_eat):
        """Handle predation between species (optimized: unified loop, reduced .any() calls)."""
        my_species = self.species

        # All directions in priority order (basic first, then extended)
        all_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),  # basic
                    (-1, -1), (-1, 1), (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]  # extended

        # Build predator mask using vectorized lookup (all species with prey can hunt)
        # Since all species can hunt all others, this is just alive & is_eat
        predator_eat = is_eat

        # Pre-generate single random tensor for all directions (reused)
        rand_attack = torch.rand((self.size, self.size), device=DEVICE)

        for dr, dc in all_dirs:
            neighbor_r = (self.rows + dr) % self.size
            neighbor_c = (self.cols + dc) % self.size
            neighbor_alive = self.alive[neighbor_r, neighbor_c]
            neighbor_species = self.species[neighbor_r, neighbor_c]
            neighbor_energy = self.energy[neighbor_r, neighbor_c]

            is_valid_prey = self._is_valid_prey_fast(my_species, neighbor_species)
            can_attack = predator_eat & neighbor_alive & is_valid_prey
            attack_success = rand_attack < 0.5
            can_eat = can_attack & attack_success
            escaped = can_attack & ~attack_success

            # Reward escaping prey (direct tensor operation, no .any() check)
            escape_prey_r = neighbor_r[escaped]
            escape_prey_c = neighbor_c[escaped]
            self.reward[escape_prey_r, escape_prey_c] += REWARD_SURVIVE_ATTACK

            # Process successful hunts (direct tensor operation, no .any() check)
            gained = neighbor_energy[can_eat] * ATTACK_BONUS
            self.energy[can_eat] = torch.clamp(self.energy[can_eat] + gained, max=MAX_ENERGY)
            self.hunger[can_eat] = 0
            self.reward[can_eat] += REWARD_EAT_PREY
            prey_r = neighbor_r[can_eat]
            prey_c = neighbor_c[can_eat]
            self.alive[prey_r, prey_c] = False
            self.energy[prey_r, prey_c] = 0
            predator_eat = predator_eat & ~can_eat

    def _parallel_reproduce(self, is_reproduce, rand_shared=None):
        """Handle reproduction with inheritance and mutation (optimized: reduced .any() calls)."""
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

        # Vectorized parameter lookups
        repro_threshold = self._get_species_param_fast(SPECIES_TENSORS['repro_threshold'])
        repro_cost = self._get_species_param_fast(SPECIES_TENSORS['repro_cost'])
        offspring_energy = self._get_species_param_fast(SPECIES_TENSORS['offspring_energy'])

        # Use shared random or generate new
        if rand_shared is None:
            rand_shared = torch.rand((self.size, self.size), device=DEVICE)

        for dr, dc in dirs:
            target_r = (self.rows + dr) % self.size
            target_c = (self.cols + dc) % self.size
            target_empty = ~self.alive[target_r, target_c]
            can_reproduce = is_reproduce & target_empty & (self.energy >= repro_threshold)
            winner = can_reproduce & (rand_shared > 0.5)

            # Direct tensor operations (empty mask = no-op, no .any() check needed)
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

    # -------------------------------------------------------------------------
    # Blob Detection and Separation
    # -------------------------------------------------------------------------
    def _find_species_blobs(self, species_id: int):
        """
        Find connected components (blobs) for a species using scipy's optimized label.
        Returns list of position tensors, one per blob, sorted by size (largest first).
        """
        species_mask = (self.species == species_id) & self.alive
        if not species_mask.any():
            return []

        # Use scipy's fast connected component labeling (8-connectivity)
        mask_np = species_mask.cpu().numpy()
        # Note: scipy doesn't handle toroidal wrapping, but for blob detection
        # this approximation is acceptable for performance
        labeled, num_blobs = ndimage.label(mask_np, structure=np.ones((3, 3)))

        if num_blobs == 0:
            return []

        blobs = []
        for blob_id in range(1, num_blobs + 1):
            positions = np.argwhere(labeled == blob_id)
            if len(positions) > 0:
                blob_tensor = torch.tensor(positions, dtype=torch.long, device=DEVICE)
                blobs.append(blob_tensor)

        # Sort by size (largest first)
        blobs.sort(key=lambda x: len(x), reverse=True)
        return blobs

    def _split_species_by_blob(self, species_id: int, new_species_id: int, blob_idx: int = 1):
        """
        Convert a specific blob of a species to a new species.
        blob_idx=1 means convert the second largest blob (keep largest as original).
        """
        blobs = self._find_species_blobs(species_id)
        if len(blobs) <= blob_idx:
            return 0  # Not enough blobs to split

        # Get the blob to convert
        blob_positions = blobs[blob_idx]
        blob_r = blob_positions[:, 0]
        blob_c = blob_positions[:, 1]

        # Convert to new species
        self.species[blob_r, blob_c] = new_species_id

        # Apply mutation to neural network weights
        mutation_w1 = torch.randn_like(self.w1[blob_r, blob_c]) * SPLIT_MUTATION_RATE
        mutation_w2 = torch.randn_like(self.w2[blob_r, blob_c]) * SPLIT_MUTATION_RATE
        self.w1[blob_r, blob_c] += mutation_w1
        self.w2[blob_r, blob_c] += mutation_w2

        return len(blob_positions)

    def _check_blob_separation(self):
        """
        Check for species with separated blobs and trigger speciation after BLOB_SEPARATION_DELAY.
        """
        if self.generation % BLOB_CHECK_INTERVAL != 0:
            return

        # Get alive species counts
        alive_species_mask = self.species[self.alive]
        if len(alive_species_mask) == 0:
            return

        counts = torch.bincount(alive_species_mask.long(), minlength=MAX_SPECIES)

        for sp_id in range(len(SPECIES_CONFIG)):
            if SPECIES_CONFIG[sp_id]['extinct']:
                continue
            if counts[sp_id].item() < 10:  # Need minimum population for blob analysis
                # Clear tracking for small populations
                if sp_id in self.blob_separation_tracker:
                    del self.blob_separation_tracker[sp_id]
                continue

            blobs = self._find_species_blobs(sp_id)

            if len(blobs) <= 1:
                # Single blob or empty - clear separation tracking
                if sp_id in self.blob_separation_tracker:
                    del self.blob_separation_tracker[sp_id]
                continue

            # Multiple blobs detected - track separation
            if sp_id not in self.blob_separation_tracker:
                self.blob_separation_tracker[sp_id] = self.generation
                continue

            # Check if separated long enough
            separation_duration = self.generation - self.blob_separation_tracker[sp_id]
            if separation_duration < BLOB_SEPARATION_DELAY:
                continue

            # Blob separation speciation!
            # The second largest blob becomes a new species
            second_blob_size = len(blobs[1])
            total_size = counts[sp_id].item()

            # Only speciate if second blob is significant (>10% of species)
            if second_blob_size < total_size * 0.1:
                continue

            # Create new species with same hidden size (geographic isolation, not mutation)
            new_species_id = create_child_species(sp_id, hidden_delta=0, current_gen=self.generation)
            if new_species_id is None:
                continue

            # Convert the second blob to new species
            num_converted = self._split_species_by_blob(sp_id, new_species_id, blob_idx=1)

            parent_name = SPECIES_CONFIG[sp_id]['name']
            new_name = SPECIES_CONFIG[new_species_id]['name']

            print(f"\n[ISOLATION] Geographic separation in {parent_name}")
            print(f"            Blob split after {separation_duration} generations apart")
            print(f"            New species: {new_name} ({num_converted} cells)")
            print(f"            Active species: {sum(1 for sp in SPECIES_CONFIG if not sp['extinct'])}/{len(SPECIES_CONFIG)}")

            # Handle history for new species
            if new_species_id < len(self.history['species']):
                self.history['species'][new_species_id] = [0] * len(self.history['population'])
            else:
                self.history['species'].append([0] * len(self.history['population']))

            # Clear separation tracker for this species (now single blob)
            del self.blob_separation_tracker[sp_id]

    def _check_random_mutation(self):
        """
        Check for random mutations every MUTATION_INTERVAL generations.
        Each alive species has RANDOM_MUTATION_CHANCE to spawn a mutant species.
        Mutants can gain or lose neurons.
        """
        if self.generation % MUTATION_INTERVAL != 0 or self.generation == 0:
            return

        # Get alive species (those with at least 1 member)
        alive_species_mask = self.species[self.alive]
        if len(alive_species_mask) == 0:
            return

        counts = torch.bincount(alive_species_mask.long(), minlength=MAX_SPECIES)

        for sp_id in range(len(SPECIES_CONFIG)):
            if counts[sp_id].item() == 0:
                continue  # Skip extinct species

            if np.random.random() > RANDOM_MUTATION_CHANCE:
                continue  # No mutation this time

            # Random hidden delta: +/- HIDDEN_SIZE_INCREMENT (with slight bias toward gain)
            if np.random.random() < 0.6:
                hidden_delta = HIDDEN_SIZE_INCREMENT  # 60% chance to gain
            else:
                hidden_delta = -HIDDEN_SIZE_INCREMENT  # 40% chance to lose

            new_species_id = create_child_species(sp_id, hidden_delta=hidden_delta, current_gen=self.generation)
            if new_species_id is None:
                continue  # Max species reached

            parent_name = SPECIES_CONFIG[sp_id]['name']
            new_name = SPECIES_CONFIG[new_species_id]['name']
            parent_hidden = SPECIES_CONFIG[sp_id]['hidden_size']
            new_hidden = SPECIES_CONFIG[new_species_id]['hidden_size']
            delta_str = f"+{hidden_delta}" if hidden_delta > 0 else str(hidden_delta)

            print(f"\n[MUTATION] Random mutation in {parent_name}")
            print(f"           New species: {new_name}")
            print(f"           Hidden neurons: {parent_hidden} -> {new_hidden} ({delta_str})")
            print(f"           Active species: {sum(1 for sp in SPECIES_CONFIG if not sp['extinct'])}/{len(SPECIES_CONFIG)}")

            # Handle history tracking for new/recycled species
            if new_species_id < len(self.history['species']):
                self.history['species'][new_species_id] = [0] * len(self.history['population'])
            else:
                self.history['species'].append([0] * len(self.history['population']))

            # Use blob-based splitting: convert entire blob to new species
            blobs = self._find_species_blobs(sp_id)
            if len(blobs) >= 2:
                # Multiple blobs: convert the second largest blob
                num_converted = self._split_species_by_blob(sp_id, new_species_id, blob_idx=1)
                print(f"           Blob of {num_converted} cells becomes {new_name}")
            elif len(blobs) == 1 and len(blobs[0]) > 1:
                # Single blob: split it in half spatially
                blob = blobs[0]
                # Use centroid to split
                centroid_r = blob[:, 0].float().mean()
                half_size = len(blob) // 5  # Take 20% of the blob
                # Sort by distance from centroid and take furthest ones
                distances = ((blob[:, 0].float() - centroid_r) ** 2 +
                            (blob[:, 1].float() - blob[:, 1].float().mean()) ** 2)
                _, indices = distances.sort(descending=True)
                split_indices = indices[:half_size]

                if len(split_indices) > 0:
                    split_r = blob[split_indices, 0]
                    split_c = blob[split_indices, 1]
                    self.species[split_r, split_c] = new_species_id
                    # Apply mutation
                    mutation_w1 = torch.randn_like(self.w1[split_r, split_c]) * SPLIT_MUTATION_RATE
                    mutation_w2 = torch.randn_like(self.w2[split_r, split_c]) * SPLIT_MUTATION_RATE
                    self.w1[split_r, split_c] += mutation_w1
                    self.w2[split_r, split_c] += mutation_w2
                    print(f"           {len(split_indices)} cells (edge of blob) become {new_name}")

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

        # Mark extinct species with extinction generation
        for sp_id, count in enumerate(species_counts):
            if count == 0 and not SPECIES_CONFIG[sp_id]['extinct']:
                SPECIES_CONFIG[sp_id]['extinct'] = True
                SPECIES_CONFIG[sp_id]['extinct_gen'] = self.generation

        new_species_id = add_new_species(dominant_species, current_gen=self.generation)
        if new_species_id is None:
            print(f"\nMax species limit ({MAX_SPECIES}) reached and no recyclable slots")
            return False

        parent_name = SPECIES_CONFIG[dominant_species]['name']
        new_name = SPECIES_CONFIG[new_species_id]['name']
        new_hidden = SPECIES_CONFIG[new_species_id]['hidden_size']
        parent_hidden = SPECIES_CONFIG[dominant_species]['hidden_size']
        print(f"\n[SPECIATION] {parent_name} at {dominance_ratio*100:.1f}% > {DOMINANCE_THRESHOLD*100}%")
        print(f"             {parent_name} splits into {new_name}")
        print(f"             Hidden neurons: {parent_hidden} -> {new_hidden}")
        print(f"             Active species: {sum(1 for sp in SPECIES_CONFIG if not sp['extinct'])}/{len(SPECIES_CONFIG)}")

        # Handle history tracking for new/recycled species
        if new_species_id < len(self.history['species']):
            # Recycled slot: reset history to zeros
            self.history['species'][new_species_id] = [0] * len(self.history['population'])
        else:
            # New slot: append new history
            self.history['species'].append([0] * len(self.history['population']))

        # Use blob-based splitting for dominance speciation
        blobs = self._find_species_blobs(dominant_species)

        if len(blobs) >= 2:
            # Multiple blobs: convert the second largest blob entirely
            num_converted = self._split_species_by_blob(dominant_species, new_species_id, blob_idx=1)
            print(f"             Blob of {num_converted} cells becomes {new_name}")
        elif len(blobs) == 1 and len(blobs[0]) > 1:
            # Single blob: split it in half spatially (by centroid distance)
            blob = blobs[0]
            centroid_r = blob[:, 0].float().mean()
            centroid_c = blob[:, 1].float().mean()

            # Sort by distance from centroid, take the far half
            distances = ((blob[:, 0].float() - centroid_r) ** 2 +
                        (blob[:, 1].float() - centroid_c) ** 2)
            _, indices = distances.sort(descending=True)
            num_to_split = len(blob) // 2
            split_indices = indices[:num_to_split]

            if len(split_indices) > 0:
                split_r = blob[split_indices, 0]
                split_c = blob[split_indices, 1]
                self.species[split_r, split_c] = new_species_id

                # Apply higher mutation during speciation
                mutation_w1 = torch.randn_like(self.w1[split_r, split_c]) * SPLIT_MUTATION_RATE
                mutation_w2 = torch.randn_like(self.w2[split_r, split_c]) * SPLIT_MUTATION_RATE
                self.w1[split_r, split_c] += mutation_w1
                self.w2[split_r, split_c] += mutation_w2

                print(f"             {len(split_indices)} cells (far half of blob) become {new_name}")
        else:
            print(f"             No cells to split")
            return False
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
    print(f"  Initial species: {INITIAL_NUM_SPECIES}, Max slots: {MAX_SPECIES}")
    print(f"  Species recycling: extinct slots reused after {EXTINCT_RECYCLE_DELAY} generations")
    print(f"  Hidden neurons: {SPECIES_HIDDEN_SIZE} -> +/-{HIDDEN_SIZE_INCREMENT}/mutation (range {MIN_HIDDEN_SIZE}-{MAX_HIDDEN_SIZE})")
    print(f"\n[Random Mutation]")
    print(f"  Check every {MUTATION_INTERVAL} generations")
    print(f"  {RANDOM_MUTATION_CHANCE*100:.0f}% chance per species per check")
    print(f"  60% gain neurons, 40% lose neurons")
    print(f"\n[Geographic Isolation]")
    print(f"  Blob separation check every {BLOB_CHECK_INTERVAL} generations")
    print(f"  Auto-speciate after {BLOB_SEPARATION_DELAY} generations apart")
    print(f"\n[Network Persistence]")
    print(f"  Save best network every {SAVE_INTERVAL} generations to: {SAVE_FILE}")
    print(f"  Fitness = lifetime + reproduction_count * 10")
    print(f"  {ELITE_RATIO*100:.0f}% of initial cells inherit saved weights")
    print(f"\n[Performance]")
    print(f"  History window: {HISTORY_WINDOW if HISTORY_WINDOW > 0 else 'unlimited'} generations")
    print(f"  Render interval: {RENDER_INTERVAL} step(s) per frame")
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

    frame_counter = [0]  # Use list for nonlocal mutation

    def update(_frame):
        nonlocal species_lines, global_species_lines

        start = time.time()
        # Run multiple steps per frame for faster simulation
        for _ in range(RENDER_INTERVAL):
            game.step()
            if game.is_extinct():
                break
        elapsed = time.time() - start

        frame_counter[0] += 1
        should_update_charts = (frame_counter[0] % CHART_UPDATE_INTERVAL == 0)

        # Add new species lines if needed (always check this)
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

        # Always update display (fast operation)
        img_display.set_array(game.render())
        ax_main.set_title(f'Digital Primordial Soup {mode_str} - Gen {game.generation} ({len(SPECIES_CONFIG)} species)')

        # Update charts only every CHART_UPDATE_INTERVAL frames (expensive operations)
        if should_update_charts:
            history_len = len(game.history['population'])
            if HISTORY_WINDOW > 0 and history_len > 0:
                # With sliding window, x starts from (current_gen - history_len + 1)
                start_gen = game.generation - history_len + 1
                gens = list(range(start_gen, game.generation + 1))
            else:
                gens = list(range(history_len))

            for sp_id in range(len(species_lines)):
                if sp_id < len(game.history['species']):
                    species_lines[sp_id].set_data(gens, game.history['species'][sp_id])
            line_total.set_data(gens, game.history['population'])

            if history_len > 100:
                ax_stats.set_xlim(gens[-100] if len(gens) >= 100 else gens[0], gens[-1] if gens else 100)
            ax_global.set_xlim(gens[0] if gens else 0, max(gens[-1] if gens else 100, 100))

            if game.history['population']:
                max_pop = max(game.history['population']) * 1.2
                ax_stats.set_ylim(0, max(max_pop, 100))
                ax_global.set_ylim(0, max(max_pop, 100))

            for sp_id in range(len(global_species_lines)):
                if sp_id < len(game.history['species']):
                    global_species_lines[sp_id].set_data(gens, game.history['species'][sp_id])
            line_global_total.set_data(gens, game.history['population'])

        # Hide extinct species from legend (only check when updating charts)
        if should_update_charts:
            legend_needs_update = False
            for sp_id in range(len(SPECIES_CONFIG)):
                is_extinct = SPECIES_CONFIG[sp_id].get('extinct', False)

                if sp_id < len(species_lines):
                    # Only update if visibility changed
                    if species_lines[sp_id].get_visible() != (not is_extinct):
                        species_lines[sp_id].set_visible(not is_extinct)
                        if is_extinct:
                            species_lines[sp_id].set_label('_hidden')
                        else:
                            name = SPECIES_CONFIG[sp_id]['name']
                            species_lines[sp_id].set_label(f'{sp_id}: {name}')
                        legend_needs_update = True

                if sp_id < len(global_species_lines):
                    if global_species_lines[sp_id].get_visible() != (not is_extinct):
                        global_species_lines[sp_id].set_visible(not is_extinct)
                        if is_extinct:
                            global_species_lines[sp_id].set_label('_hidden')
                        else:
                            name = SPECIES_CONFIG[sp_id]['name']
                            global_species_lines[sp_id].set_label(name)
                        legend_needs_update = True

            if legend_needs_update:
                ax_stats.legend(loc='upper right', fontsize=7)
                ax_global.legend(loc='upper right', fontsize=8, ncol=min(len(SPECIES_CONFIG), 5))

        # Update info text (only show alive species)
        total = game.history['population'][-1] if game.history['population'] else 0
        alive_species_count = sum(1 for sp in SPECIES_CONFIG if not sp.get('extinct', False))
        info = f"Gen: {game.generation} | Species: {alive_species_count} | Pop: {total}\n"
        for sp_id in range(len(SPECIES_CONFIG)):
            if SPECIES_CONFIG[sp_id].get('extinct', False):
                continue  # Skip extinct species entirely
            if sp_id < len(game.history['species']) and game.history['species'][sp_id]:
                count = game.history['species'][sp_id][-1]
            else:
                count = 0
            name = SPECIES_CONFIG[sp_id]['name']
            if count == 0:
                SPECIES_CONFIG[sp_id]['extinct'] = True
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
