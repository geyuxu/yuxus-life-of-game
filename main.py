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
import threading
import queue

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
GRID_SIZE = 100 #ARGS.grid
INITIAL_ENERGY = 30.0
MAX_ENERGY = 100.0
MOVE_COST = 0.2
CROWDING_THRESHOLD = 4
CROWDING_PENALTY = 0.5

# Species configuration
INITIAL_NUM_SPECIES = 3
MAX_SPECIES = 500
MAX_ACTIVE_SPECIES = 50  # Maximum number of active (non-extinct) species to prevent performance issues
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
MUTATION_INTERVAL = 100     # Check for random mutation every N generations (increased to reduce split frequency)
RANDOM_MUTATION_CHANCE = 0.05  # Probability per species per check (reduced to slow down speciation)

# Species recycling
EXTINCT_RECYCLE_DELAY = 50  # Generations before extinct species slot can be reused

# Blob-based speciation
BLOB_SEPARATION_DELAY = 0   # Generations of separation before blob becomes new species (0 = immediate)
BLOB_CHECK_INTERVAL = 200   # Check for blob separation every N generations (increased to reduce overhead)

# Performance tuning
HISTORY_WINDOW = 1000       # Keep only last N generations in history (0 = unlimited)
RENDER_INTERVAL = 1         # Render every N frames (increase for faster simulation)
CHART_UPDATE_INTERVAL = 5   # Update charts every N frames (reduces matplotlib overhead)
VERBOSE_SPECIATION = False  # Print detailed logs for each speciation event (can slow down performance)

# Combat
ATTACK_BONUS = 1.2  # Energy multiplier when eating prey

# Chemical signaling (for cell differentiation)
NUM_CHEMICALS = 4           # Number of chemical types
CHEMICAL_DIFFUSION = 0.3    # Diffusion rate (0-1)
CHEMICAL_DECAY = 0.05       # Decay rate per step
CHEMICAL_SECRETION = 0.1    # Amount secreted per step
CHEMICAL_INPUT_WEIGHT = 0.2 # How much chemicals affect NN input

# Evolution
MUTATION_RATE = 0.1
DOMINANCE_THRESHOLD = 0.75  # Species split when >75% of population
SPLIT_MUTATION_RATE = 0.3   # Higher mutation during speciation

# Genome-based visualization (Scheme C: Hybrid Genome)
GENOME_BASED_COLOR = True   # Enable genome-to-color mapping (similar genomes = similar colors)
GENOME_COLOR_UPDATE_INTERVAL = 200  # Update species colors every N generations (increased to reduce overhead)

# Reinforcement Learning
RL_LEARNING_RATE = 0.01
REWARD_EAT_PREY = 2.0
REWARD_SURVIVE_ATTACK = 1.0
REWARD_REPRODUCE = 1.5

# Neural Network Persistence
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FILE = os.path.join(SAVE_DIR, "best_brain.pt")
SAVE_INTERVAL = 50
AUTO_SAVE_ENABLED = False  # Auto-save disabled by default, use manual save instead (press 'S' in pygame)
ELITE_RATIO = 0.2

# Neural Network Architecture
INPUT_SIZE = 20 + NUM_CHEMICALS  # Original 20 + chemical concentrations
NUM_ACTIONS = 7
ACTION_STAY = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_EAT = 5
ACTION_REPRODUCE = 6

# =============================================================================
# GENOME-BASED IDENTITY (No Species IDs)
# =============================================================================
# All cells share the same base parameters
# Individual identity comes from genome (12-dim fingerprint)

# Genome-based mate finding
MATE_GENOME_THRESHOLD = 0.5  # Max genome distance for compatible mates (lower = more selective)


# =============================================================================
# ASYNC WEIGHT SAVER
# =============================================================================
class AsyncWeightSaver:
    """Asynchronous weight saver to avoid blocking the main thread during torch.save()."""

    def __init__(self):
        self.save_queue = queue.Queue(maxsize=2)  # Limit queue size to avoid memory issues
        self.worker_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.worker_thread.start()
        self.saving = False

    def _save_worker(self):
        """Background worker that processes save requests."""
        while True:
            checkpoint, filepath = self.save_queue.get()
            if checkpoint is None:  # Sentinel to stop thread
                break
            try:
                self.saving = True
                torch.save(checkpoint, filepath)
                print(f"[SAVED] Fitness={checkpoint['fitness']:.1f}, Hidden={checkpoint['hidden_size']}, Gen {checkpoint['generation']}")
            except Exception as e:
                print(f"[ERROR] Failed to save weights: {e}")
            finally:
                self.saving = False
                self.save_queue.task_done()

    def save_async(self, checkpoint, filepath):
        """Queue a checkpoint for asynchronous saving."""
        try:
            # Try to add to queue without blocking
            # If queue is full, skip this save (previous save still in progress)
            self.save_queue.put_nowait((checkpoint, filepath))
        except queue.Full:
            print("[SKIP] Save queue full, skipping this save")

    def is_saving(self):
        """Check if a save operation is in progress."""
        return self.saving or not self.save_queue.empty()

    def shutdown(self):
        """Shutdown the saver thread gracefully."""
        self.save_queue.put((None, None))
        self.worker_thread.join(timeout=5.0)


# Global async saver instance
ASYNC_SAVER = AsyncWeightSaver()


# =============================================================================
# GENETIC GENOME SYSTEM (Scheme C: Hybrid Genome)
# =============================================================================

def neural_fingerprint(w1: torch.Tensor, w2: torch.Tensor, hidden_size: int) -> np.ndarray:
    """
    Extract statistical fingerprint from neural network weights.

    Returns 8-dimensional feature vector:
    - w1 features (4-dim): mean, std, abs_mean, max
    - w2 features (4-dim): mean, std, abs_mean, max

    Args:
        w1: First layer weights [hidden_size, INPUT_SIZE]
        w2: Second layer weights [NUM_ACTIONS, hidden_size]
        hidden_size: Hidden layer size

    Returns:
        8-dimensional numpy array representing neural fingerprint
    """
    # Move to CPU and convert to numpy for consistency
    w1_np = w1.cpu().numpy() if isinstance(w1, torch.Tensor) else w1
    w2_np = w2.cpu().numpy() if isinstance(w2, torch.Tensor) else w2

    # Extract w1 features
    w1_mean = np.mean(w1_np)
    w1_std = np.std(w1_np)
    w1_abs_mean = np.mean(np.abs(w1_np))
    w1_max = np.max(np.abs(w1_np))

    # Extract w2 features
    w2_mean = np.mean(w2_np)
    w2_std = np.std(w2_np)
    w2_abs_mean = np.mean(np.abs(w2_np))
    w2_max = np.max(np.abs(w2_np))

    return np.array([
        w1_mean, w1_std, w1_abs_mean, w1_max,
        w2_mean, w2_std, w2_abs_mean, w2_max
    ], dtype=np.float32)


def get_full_genome(w1: torch.Tensor, w2: torch.Tensor,
                    hidden_size: int, chemical_affinity: list) -> np.ndarray:
    """
    Compute hybrid genome combining neural fingerprint and chemical affinity.

    Returns 12-dimensional genome vector:
    - First 8 dimensions: neural fingerprint (genotype)
    - Last 4 dimensions: chemical affinity (phenotype)

    Args:
        w1: First layer weights
        w2: Second layer weights
        hidden_size: Hidden layer size
        chemical_affinity: Chemical affinity vector (4-dim)

    Returns:
        12-dimensional numpy array representing full genome
    """
    neural = neural_fingerprint(w1, w2, hidden_size)
    chemical = np.array(chemical_affinity, dtype=np.float32)

    return np.concatenate([neural, chemical])


def genome_to_color(genome: np.ndarray) -> tuple:
    """
    Map genome to RGB color using HSV color space.

    Combines neural fingerprint AND chemical affinity for color diversity:
    - Hue: Based on neural w1_mean + chemical affinity[0]
    - Saturation: Based on neural w1_std + chemical affinity[1]
    - Value: Based on neural w1_abs_mean + chemical affinity[2]

    Args:
        genome: 12-dimensional genome vector (8 neural + 4 chemical)

    Returns:
        RGB tuple (3 floats in range 0-1)
    """
    # Neural features (first 8 dimensions)
    w1_mean = genome[0]
    w1_std = genome[1]
    w1_abs_mean = genome[2]

    # Chemical features (last 4 dimensions)
    chem0 = genome[8]
    chem1 = genome[9]
    chem2 = genome[10]

    # Map to HSV with normalization
    # Hue: Combine neural mean + chemical[0] for more diversity
    hue_neural = np.tanh(w1_mean)
    hue_chem = np.tanh(chem0 * 2.0)  # Scale chemical influence
    hue = ((hue_neural + hue_chem) / 2.0 + 1.0) / 2.0  # Average and normalize to [0, 1]

    # Saturation: Combine neural std + chemical[1]
    sat_neural = np.tanh(w1_std * 2.0)
    sat_chem = np.tanh(chem1 * 2.0)
    saturation = 0.5 + 0.5 * (sat_neural + sat_chem) / 2.0  # Range [0.5, 1.0]

    # Value (brightness): Combine neural abs_mean + chemical[2]
    val_neural = np.tanh(w1_abs_mean)
    val_chem = np.tanh(chem2 * 2.0)
    value = 0.6 + 0.4 * (val_neural + val_chem) / 2.0  # Range [0.6, 1.0]

    return colorsys.hsv_to_rgb(hue, saturation, value)


def genetic_distance(genome1: np.ndarray, genome2: np.ndarray) -> float:
    """
    Calculate genetic distance between two genomes using Euclidean distance.

    Args:
        genome1: First genome (12-dim)
        genome2: Second genome (12-dim)

    Returns:
        Euclidean distance between genomes
    """
    return np.linalg.norm(genome1 - genome2)


def genetic_similarity(genome1: np.ndarray, genome2: np.ndarray) -> float:
    """
    Calculate genetic similarity as normalized inverse distance.

    Returns value in range [0, 1] where 1 = identical genomes.

    Args:
        genome1: First genome (12-dim)
        genome2: Second genome (12-dim)

    Returns:
        Similarity score (0-1)
    """
    distance = genetic_distance(genome1, genome2)
    # Use exponential decay for smooth similarity curve
    # sigma=2.0 means similarity drops to ~0.6 at distance=1.0
    return np.exp(-distance**2 / (2 * 2.0**2))


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
        self.hunger = torch.zeros((size, size), dtype=torch.int32, device=DEVICE)
        self.is_newborn = torch.zeros((size, size), dtype=torch.bool, device=DEVICE)

        # Chemical signaling field [NUM_CHEMICALS, size, size]
        self.chemicals = torch.zeros((NUM_CHEMICALS, size, size), dtype=torch.float32, device=DEVICE)

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

        # Genome (12-dim fingerprint: 8 neural + 4 chemical)
        self.genome = torch.zeros((size, size, 12), dtype=torch.float32, device=DEVICE)

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

        self.history = {'population': []}

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

                # Adapt weights to match current INPUT_SIZE and MAX_HIDDEN_SIZE
                saved_input_size = w1.shape[0]
                saved_hidden = w1.shape[1]

                # Create new w1 with current dimensions
                new_w1 = torch.randn((INPUT_SIZE, MAX_HIDDEN_SIZE), device=DEVICE) * 0.1

                # Copy overlapping region from saved weights
                copy_input = min(saved_input_size, INPUT_SIZE)
                copy_hidden = min(saved_hidden, MAX_HIDDEN_SIZE)
                new_w1[:copy_input, :copy_hidden] = w1[:copy_input, :copy_hidden].to(DEVICE)

                if saved_input_size != INPUT_SIZE or saved_hidden != MAX_HIDDEN_SIZE:
                    print(f"  Adapting w1: [{saved_input_size}, {saved_hidden}] -> [{INPUT_SIZE}, {MAX_HIDDEN_SIZE}]")

                # Adapt w2: [saved_hidden, NUM_ACTIONS] -> [MAX_HIDDEN_SIZE, NUM_ACTIONS]
                new_w2 = torch.randn((MAX_HIDDEN_SIZE, NUM_ACTIONS), device=DEVICE) * 0.1
                new_w2[:copy_hidden, :] = w2[:copy_hidden, :].to(DEVICE)

                if saved_hidden != MAX_HIDDEN_SIZE:
                    print(f"  Adapting w2: [{saved_hidden}, {NUM_ACTIONS}] -> [{MAX_HIDDEN_SIZE}, {NUM_ACTIONS}]")

                return new_w1, new_w2, hidden_size
            except Exception as e:
                print(f"Warning: Failed to load weights: {e}")
                return None, None, SPECIES_HIDDEN_SIZE
        else:
            print("No saved weights found, using random initialization")
            return None, None, SPECIES_HIDDEN_SIZE

    def _save_best_weights(self):
        """Save the current best neural network weights and structure asynchronously."""
        if self.best_w1 is None:
            return

        # Skip if previous save still in progress
        if ASYNC_SAVER.is_saving():
            print(f"[SKIP] Previous save in progress, skipping gen {self.generation}")
            return

        # Prepare checkpoint (move to CPU to avoid GPU memory issues)
        checkpoint = {
            'w1': self.best_w1.cpu().clone(),
            'w2': self.best_w2.cpu().clone(),
            'hidden_size': self.best_hidden_size,
            'fitness': self.best_fitness,
            'generation': self.generation,
        }

        # Queue for async save (non-blocking)
        ASYNC_SAVER.save_async(checkpoint, SAVE_FILE)

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
            # Use default hidden size (all cells share same architecture)
            self.best_hidden_size = SPECIES_HIDDEN_SIZE

    # -------------------------------------------------------------------------
    # Genome Computation (Scheme C: Hybrid Genome)
    # -------------------------------------------------------------------------
    def compute_genome(self, y: int, x: int) -> np.ndarray:
        """
        Compute hybrid genome for a cell at position (y, x).

        Returns 12-dimensional genome combining:
        - Neural fingerprint (8-dim): Statistical features from w1, w2
        - Chemical affinity (4-dim): From self.genome (already stored)

        Args:
            y: Row position
            x: Column position

        Returns:
            12-dimensional numpy array
        """
        if not self.alive[y, x]:
            return None

        # Get neural network weights for this cell
        w1_cell = self.w1[y, x]  # [INPUT_SIZE, MAX_HIDDEN_SIZE]
        w2_cell = self.w2[y, x]  # [MAX_HIDDEN_SIZE, NUM_ACTIONS]

        # Chemical affinity is already stored in self.genome (last 4 dimensions)
        chemical_affinity = self.genome[y, x, 8:12].cpu().numpy()

        # Compute hybrid genome
        return get_full_genome(w1_cell, w2_cell, SPECIES_HIDDEN_SIZE, chemical_affinity)

    def compute_all_genomes(self) -> dict:
        """
        Compute genomes for all alive cells.

        Returns:
            Dictionary mapping (y, x) positions to genome vectors
        """
        genomes = {}
        alive_positions = self.alive.nonzero(as_tuple=False)

        for pos in alive_positions:
            y, x = pos[0].item(), pos[1].item()
            genome = self.compute_genome(y, x)
            if genome is not None:
                genomes[(y, x)] = genome

        return genomes


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

        # Initialize genome with spatial diversity
        # Chemical affinity (last 4 dims) varies by position for initial diversity
        for i in range(num_cells):
            r, c = rows[i].item(), cols[i].item()
            # Position-based seed for reproducible but varied initialization
            position_seed = r * self.size + c
            torch.manual_seed(position_seed + 1000)
            # Chemical affinity: position-dependent randomization
            self.genome[r, c, 8:12] = torch.randn(NUM_CHEMICALS, device=DEVICE) * 0.5

        # Load saved weights for elite cells (50% saved + 50% random for diversity)
        if self.saved_w1 is not None and self.saved_w2 is not None:
            num_elite = int(num_cells * ELITE_RATIO)
            if num_elite > 0:
                elite_indices = torch.randperm(num_cells, device=DEVICE)[:num_elite]
                elite_rows = rows[elite_indices]
                elite_cols = cols[elite_indices]

                # Only copy 50% of hidden neurons from saved weights
                # The rest remain random-initialized for diversity
                saved_hidden = self.saved_hidden_size
                copy_hidden = saved_hidden // 2  # Only use half of saved neurons

                # w1: [INPUT_SIZE, MAX_HIDDEN_SIZE]
                # Copy only first half of hidden neurons from saved weights
                self.w1[elite_rows, elite_cols, :, :copy_hidden] = \
                    self.saved_w1[:, :copy_hidden].clone() + torch.randn((INPUT_SIZE, copy_hidden), device=DEVICE) * 0.05

                # w2: [MAX_HIDDEN_SIZE, NUM_ACTIONS]
                # Copy only first half of hidden neurons from saved weights
                self.w2[elite_rows, elite_cols, :copy_hidden, :] = \
                    self.saved_w2[:copy_hidden, :].clone() + torch.randn((copy_hidden, NUM_ACTIONS), device=DEVICE) * 0.05

                # The rest of hidden neurons (copy_hidden:MAX_HIDDEN_SIZE) remain random
                # This creates diversity: each cell has 50% learned + 50% random
                print(f"{num_elite} cells inherited saved weights (50% saved + 50% random for diversity)")

        # Update genome neural fingerprints based on initialized weights
        for i in range(num_cells):
            r, c = rows[i].item(), cols[i].item()
            genome_np = self.compute_genome(r, c)
            if genome_np is not None:
                self.genome[r, c] = torch.from_numpy(genome_np).float().to(DEVICE)

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

        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        neighbor_energy = torch.stack([self._get_shifted(energy_norm * alive_f, dr, dc) for dr, dc in dirs], dim=-1)
        neighbor_alive = torch.stack([self._get_shifted(alive_f, dr, dc) for dr, dc in dirs], dim=-1)

        # Compute genome-based similarity: use chemical affinity (last 4 dims of genome)
        # for efficient similarity calculation
        chem_affinity = self.genome[:, :, 8:12]  # [H, W, 4]
        similar_count = torch.zeros((self.size, self.size, 1), device=DEVICE)

        for dr, dc in dirs:
            neighbor_chem = self._get_shifted(chem_affinity, dr, dc)  # [H, W, 4]
            neighbor_is_alive = self._get_shifted(alive_f, dr, dc)  # [H, W]
            # Compute distance in chemical affinity space
            distance = torch.norm(chem_affinity - neighbor_chem, dim=-1)  # [H, W]
            is_similar = (distance < MATE_GENOME_THRESHOLD).float() * neighbor_is_alive
            similar_count[:, :, 0] += is_similar

        total_neighbors = neighbor_alive.sum(dim=-1, keepdim=True)
        diff_count = total_neighbors - similar_count

        # Local chemical concentrations (transpose for correct indexing)
        local_chemicals = self.chemicals[:, :, :].permute(1, 2, 0)  # [H, W, NUM_CHEMICALS]

        inputs = torch.cat([
            neighbor_energy,                                    # 8: neighbor energy levels
            similar_count / 8.0,                                # 1: similar genome count (was same species)
            diff_count / 8.0,                                   # 1: different genome count
            energy_norm.unsqueeze(-1),                          # 1: own energy
            total_neighbors / 8.0,                              # 1: total neighbor count
            torch.zeros((self.size, self.size, 8), device=DEVICE),  # 8: padding
            local_chemicals * CHEMICAL_INPUT_WEIGHT,            # NUM_CHEMICALS: local chemical signals
        ], dim=-1)
        return inputs

    def _batch_forward(self, inputs):
        """Batch forward pass through all cells' neural networks (CUDA optimized)."""
        h = torch.tanh(torch.einsum('ijk,ijkl->ijl', inputs, self.w1))

        # Apply hidden layer mask (all cells share same architecture)
        mask = (self.neuron_idx < SPECIES_HIDDEN_SIZE).float()
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

    def _diffuse_chemicals(self):
        """Diffuse chemicals across the grid using convolution."""
        # Create diffusion kernel (3x3 average with center weight)
        kernel = torch.tensor([[
            [0.05, 0.1, 0.05],
            [0.1,  0.4, 0.1],
            [0.05, 0.1, 0.05]
        ]], dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1, 1, 3, 3]

        for chem_id in range(NUM_CHEMICALS):
            # Pad with circular boundary (toroidal world)
            chem_field = self.chemicals[chem_id:chem_id+1].unsqueeze(0)  # [1, 1, H, W]
            padded = F.pad(chem_field, (1, 1, 1, 1), mode='circular')

            # Convolve for diffusion
            diffused = F.conv2d(padded, kernel).squeeze()

            # Mix old and new based on diffusion rate
            self.chemicals[chem_id] = (
                (1 - CHEMICAL_DIFFUSION) * self.chemicals[chem_id] +
                CHEMICAL_DIFFUSION * diffused
            )

        # Chemical decay
        self.chemicals *= (1 - CHEMICAL_DECAY)

    def _secrete_chemicals(self):
        """Cells secrete chemicals based on their genome's chemical affinity."""
        if not self.alive.any():
            return

        # Each cell secretes based on its genome (last 4 dimensions = chemical affinity)
        # genome[:, :, 8:12] = [H, W, NUM_CHEMICALS]
        chem_affinity = self.genome[:, :, 8:12]  # [H, W, 4]

        for chem_id in range(NUM_CHEMICALS):
            affinity_map = chem_affinity[:, :, chem_id]  # [H, W]
            # Positive affinity = secretion, negative = absorption/none
            secretion_map = torch.where(
                self.alive & (affinity_map > 0),
                CHEMICAL_SECRETION * affinity_map,
                torch.zeros_like(affinity_map)
            )
            self.chemicals[chem_id] += secretion_map

        # Clamp to prevent overflow
        self.chemicals.clamp_(0, 10.0)

    def step(self):
        """Execute one simulation step (CUDA optimized)."""
        self.is_newborn.fill_(False)

        # Chemical diffusion and secretion
        self._diffuse_chemicals()
        self._secrete_chemicals()

        # Metabolism (all cells share same metabolism rate)
        self.energy = torch.where(self.alive, self.energy - SPECIES_METABOLISM, self.energy)

        # Crowding penalty (already optimized with conv2d)
        alive_f = self.alive.float()
        padded = F.pad(alive_f.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='circular')
        neighbors = F.conv2d(padded, self.crowding_kernel).squeeze()
        crowding_cost = torch.clamp(neighbors - CROWDING_THRESHOLD, min=0) * CROWDING_PENALTY
        self.energy = torch.where(self.alive, self.energy - crowding_cost, self.energy)

        # Starvation (all cells share same starvation limit)
        self.hunger = torch.where(self.alive, self.hunger + 1, self.hunger)
        self.energy = torch.where(self.hunger >= SPECIES_STARVATION, torch.zeros_like(self.energy), self.energy)

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

        # Save best network periodically (only if auto-save enabled)
        self._update_best_network()
        if AUTO_SAVE_ENABLED and self.generation > 0 and self.generation % SAVE_INTERVAL == 0:
            self._save_best_weights()

        self.generation += 1

        # Statistics (simple population count)
        total = self.alive.sum().item()
        self.history['population'].append(total)

        # Sliding window: trim old history to reduce memory
        if HISTORY_WINDOW > 0 and len(self.history['population']) > HISTORY_WINDOW:
            trim = len(self.history['population']) - HISTORY_WINDOW
            self.history['population'] = self.history['population'][trim:]

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
            move_hunger = self.hunger[winner]
            move_w1 = self.w1[winner]
            move_w2 = self.w2[winner]
            move_genome = self.genome[winner]

            self.alive[winner] = False
            self.energy[winner] = 0

            new_r = target_r[winner]
            new_c = target_c[winner]
            self.alive[new_r, new_c] = True
            self.energy[new_r, new_c] = move_energy
            self.hunger[new_r, new_c] = move_hunger
            self.w1[new_r, new_c] = move_w1
            self.w2[new_r, new_c] = move_w2
            self.genome[new_r, new_c] = move_genome

        # Eating (always call, method handles empty mask efficiently)
        is_eat = self.alive & (actions == ACTION_EAT)
        self._parallel_eat(is_eat)

        # Reproduction (always call, method handles empty mask efficiently)
        is_reproduce = self.alive & (actions == ACTION_REPRODUCE) & (self.energy >= SPECIES_REPRO_THRESHOLD)
        self._parallel_reproduce(is_reproduce, rand_shared)

    def _parallel_eat(self, is_eat):
        """Handle predation based on genome dissimilarity."""
        # All directions in priority order (basic first, then extended)
        all_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),  # basic
                    (-1, -1), (-1, 1), (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]  # extended

        predator_eat = is_eat

        # Pre-generate single random tensor for all directions (reused)
        rand_attack = torch.rand((self.size, self.size), device=DEVICE)

        # Get chemical affinity for genome-based prey detection
        my_chem = self.genome[:, :, 8:12]  # [H, W, 4]

        for dr, dc in all_dirs:
            neighbor_r = (self.rows + dr) % self.size
            neighbor_c = (self.cols + dc) % self.size
            neighbor_alive = self.alive[neighbor_r, neighbor_c]
            neighbor_chem = my_chem[neighbor_r, neighbor_c]  # [H, W, 4]
            neighbor_energy = self.energy[neighbor_r, neighbor_c]

            # Can eat if genome is sufficiently different (not same "kind")
            genome_dist = torch.norm(my_chem - neighbor_chem, dim=-1)  # [H, W]
            is_valid_prey = (genome_dist >= MATE_GENOME_THRESHOLD)  # Opposite of mating criterion
            can_attack = predator_eat & neighbor_alive & is_valid_prey

            # Dynamic combat success based on chemical "strength"
            # Cells with stronger local chemical fields have higher attack/defense
            predator_strength = self.chemicals.sum(dim=0)  # [H, W] - sum of all chemicals
            prey_strength = predator_strength[neighbor_r, neighbor_c]
            attacker_strength = predator_strength

            # Success probability: sigmoid((attack - defense) / 2) scaled to 0.1-0.9 range
            # This ensures: stronger attackers win more, but never guaranteed
            strength_diff = attacker_strength - prey_strength
            success_prob = torch.sigmoid(strength_diff / 2.0) * 0.8 + 0.1  # Range: 0.1 to 0.9
            success_prob = torch.clamp(success_prob, 0.1, 0.9)

            # Apply success probability
            attack_success = rand_attack < success_prob
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
        """
        Handle reproduction with genome-based mate finding and sexual crossover.

        Sexual reproduction (crossover):
        - Find a mate (similar genome neighbor, distance < MATE_GENOME_THRESHOLD)
        - Offspring inherits: neurons (50/50 from parents) + genome (50/50 from parents)
        - Apply mutation after inheritance

        Asexual reproduction (fallback):
        - If no compatible mate found, clone parent with mutation
        """
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

        # Use shared random or generate new
        if rand_shared is None:
            rand_shared = torch.rand((self.size, self.size), device=DEVICE)

        for dr, dc in dirs:
            target_r = (self.rows + dr) % self.size
            target_c = (self.cols + dc) % self.size
            target_empty = ~self.alive[target_r, target_c]
            can_reproduce = is_reproduce & target_empty & (self.energy >= SPECIES_REPRO_THRESHOLD)
            winner = can_reproduce & (rand_shared > 0.5)

            if not winner.any():
                continue

            # Direct tensor operations
            self.energy[winner] -= SPECIES_REPRO_COST
            self.reward[winner] += REWARD_REPRODUCE
            self.repro_count[winner] += 1

            child_r = target_r[winner]
            child_c = target_c[winner]
            self.alive[child_r, child_c] = True
            self.energy[child_r, child_c] = SPECIES_OFFSPRING_ENERGY
            self.hunger[child_r, child_c] = 0
            self.is_newborn[child_r, child_c] = True

            # Sexual reproduction: find mates (similar genome neighbors)
            parent1_r, parent1_c = self.rows[winner], self.cols[winner]
            parent1_genome = self.genome[parent1_r, parent1_c]  # [N, 12]
            num_parents = len(parent1_r)

            # Search for mate in 8 neighbors
            mate_found = torch.zeros(num_parents, dtype=torch.bool, device=DEVICE)
            mate_r = torch.zeros(num_parents, dtype=torch.long, device=DEVICE)
            mate_c = torch.zeros(num_parents, dtype=torch.long, device=DEVICE)

            for mate_dr, mate_dc in dirs:
                if mate_found.all():
                    break

                candidate_r = (parent1_r + mate_dr) % self.size
                candidate_c = (parent1_c + mate_dc) % self.size

                is_alive = self.alive[candidate_r, candidate_c]
                candidate_genome = self.genome[candidate_r, candidate_c]  # [N, 12]

                # Genome compatibility: distance < threshold (similar genomes can mate)
                genome_dist = torch.norm(parent1_genome - candidate_genome, dim=-1)  # [N]
                is_compatible = (genome_dist < MATE_GENOME_THRESHOLD)
                is_valid_mate = is_alive & is_compatible & ~mate_found

                mate_r = torch.where(is_valid_mate, candidate_r, mate_r)
                mate_c = torch.where(is_valid_mate, candidate_c, mate_c)
                mate_found = mate_found | is_valid_mate

            # Get parent weights and genomes
            parent1_w1 = self.w1[parent1_r, parent1_c]
            parent1_w2 = self.w2[parent1_r, parent1_c]

            # Sexual reproduction: crossover from two parents
            sexual_mask = mate_found
            if sexual_mask.any():
                sexual_indices = sexual_mask.nonzero(as_tuple=True)[0]
                num_sexual = len(sexual_indices)

                parent2_w1 = self.w1[mate_r[sexual_indices], mate_c[sexual_indices]]
                parent2_w2 = self.w2[mate_r[sexual_indices], mate_c[sexual_indices]]
                parent2_genome = self.genome[mate_r[sexual_indices], mate_c[sexual_indices]]

                # Create crossover mask: random 50/50 selection for neurons
                crossover_mask_w1 = torch.rand((num_sexual, INPUT_SIZE, MAX_HIDDEN_SIZE), device=DEVICE) > 0.5
                crossover_mask_w2 = torch.rand((num_sexual, MAX_HIDDEN_SIZE, NUM_ACTIONS), device=DEVICE) > 0.5
                crossover_mask_genome = torch.rand((num_sexual, 12), device=DEVICE) > 0.5

                # Crossover: interleave neurons and genome from both parents
                child_w1_sexual = torch.where(crossover_mask_w1, parent1_w1[sexual_indices], parent2_w1)
                child_w2_sexual = torch.where(crossover_mask_w2, parent1_w2[sexual_indices], parent2_w2)
                child_genome_sexual = torch.where(crossover_mask_genome, parent1_genome[sexual_indices], parent2_genome)

                # Apply mutation
                child_w1_sexual += torch.randn_like(child_w1_sexual) * MUTATION_RATE
                child_w2_sexual += torch.randn_like(child_w2_sexual) * MUTATION_RATE
                child_genome_sexual += torch.randn_like(child_genome_sexual) * MUTATION_RATE * 0.1  # Smaller mutation for genome

                # Assign to offspring
                self.w1[child_r[sexual_indices], child_c[sexual_indices]] = child_w1_sexual
                self.w2[child_r[sexual_indices], child_c[sexual_indices]] = child_w2_sexual
                self.genome[child_r[sexual_indices], child_c[sexual_indices]] = child_genome_sexual

            # Asexual reproduction: clone parent (fallback for cells without mate)
            asexual_mask = ~mate_found
            if asexual_mask.any():
                asexual_indices = asexual_mask.nonzero(as_tuple=True)[0]

                # Simple cloning with mutation
                self.w1[child_r[asexual_indices], child_c[asexual_indices]] = \
                    parent1_w1[asexual_indices] + torch.randn_like(parent1_w1[asexual_indices]) * MUTATION_RATE
                self.w2[child_r[asexual_indices], child_c[asexual_indices]] = \
                    parent1_w2[asexual_indices] + torch.randn_like(parent1_w2[asexual_indices]) * MUTATION_RATE
                self.genome[child_r[asexual_indices], child_c[asexual_indices]] = \
                    parent1_genome[asexual_indices] + torch.randn_like(parent1_genome[asexual_indices]) * MUTATION_RATE * 0.1

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
    def render(self):
        """Render simulation state as RGB image with genome-based colors."""
        img = np.zeros((self.size, self.size, 3), dtype=np.float32)

        alive = self.alive.cpu().numpy()
        energy = self.energy.cpu().numpy()
        genome = self.genome.cpu().numpy()
        newborn = self.is_newborn.cpu().numpy()

        energy_norm = np.clip(energy / MAX_ENERGY, 0, 1)

        # Render each alive cell with genome-based color
        for y in range(self.size):
            for x in range(self.size):
                if alive[y, x]:
                    base_color = genome_to_color(genome[y, x])
                    brightness = 0.3 + 0.7 * energy_norm[y, x]
                    img[y, x] = np.array(base_color) * brightness

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
    print("Digital Primordial Soup - Genome-Based Evolution")
    print("=" * 60)
    print(f"\n[Genome-Based Identity System]")
    print("  No fixed species - identity is fluid based on 12-dim genome")
    print("  Genome = Neural fingerprint (8-dim) + Chemical affinity (4-dim)")
    print("  Colors reflect genome similarity")
    print("  Mating based on genome distance (threshold: {:.2f})".format(MATE_GENOME_THRESHOLD))
    print(f"\n[Shared Parameters]")
    print(f"  Metabolism: {SPECIES_METABOLISM}")
    print(f"  Reproduction threshold: {SPECIES_REPRO_THRESHOLD}")
    print(f"  Reproduction cost: {SPECIES_REPRO_COST}")
    print(f"  Offspring energy: {SPECIES_OFFSPRING_ENERGY}")
    print(f"  Starvation threshold: {SPECIES_STARVATION}")
    print(f"  Hidden layer size: {SPECIES_HIDDEN_SIZE}")
    print(f"\n[Chemical Signaling System]")
    print(f"  Number of chemicals: {NUM_CHEMICALS}")
    print(f"  Diffusion rate: {CHEMICAL_DIFFUSION}")
    print(f"  Decay rate: {CHEMICAL_DECAY}")
    print(f"  Secretion rate: {CHEMICAL_SECRETION}")
    print(f"  Each species has unique chemical affinity (evolvable)")
    print(f"  Combat success: 10%-90% based on local chemical strength")
    print(f"  Stronger chemical field = better attack/defense")
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
    if BLOB_SEPARATION_DELAY > 0:
        print(f"  Auto-speciate after {BLOB_SEPARATION_DELAY} generations apart")
    else:
        print(f"  Auto-speciate immediately when separated blobs detected")
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
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 2], width_ratios=[2, 1, 1],
                          hspace=0.15, wspace=0.15)

    # Top left: Global statistics (full history)
    ax_global = fig.add_subplot(gs[0, :2])
    global_species_lines = []
    for sp_id in range(len(SPECIES_CONFIG)):
        color = SPECIES_CONFIG[sp_id]['color']
        name = SPECIES_CONFIG[sp_id]['name']
        line, = ax_global.plot([], [], color=color, label=name, linewidth=1.5)
        global_species_lines.append(line)
    line_global_total, = ax_global.plot([], [], 'k--', label='Total', linewidth=2)
    ax_global.set_xlim(0, 100)
    ax_global.set_ylim(0, game.size * game.size * 0.15)
    ax_global.set_xlabel('Generation', fontsize=9)
    ax_global.set_ylabel('Population', fontsize=9)
    ax_global.grid(True, alpha=0.3)
    ax_global.set_title('Global Ecosystem Dynamics (Full History)', fontsize=10)
    ax_global.legend().set_visible(False)  # Hide legend, will show in right panel

    # Bottom left: Main display (simulation environment)
    ax_main = fig.add_subplot(gs[1, 0])
    img_display = ax_main.imshow(game.render(), interpolation='nearest')
    ax_main.set_title(f'Environment - Gen 0', fontsize=10)
    ax_main.axis('off')

    # Bottom middle: Recent species dynamics (last 100 generations)
    ax_recent = fig.add_subplot(gs[1, 1])
    recent_species_lines = []
    for sp_id in range(len(SPECIES_CONFIG)):
        color = SPECIES_CONFIG[sp_id]['color']
        line, = ax_recent.plot([], [], color=color, linewidth=1.5)
        recent_species_lines.append(line)
    line_recent_total, = ax_recent.plot([], [], 'k--', label='Total', linewidth=2)
    ax_recent.set_xlim(0, 100)
    ax_recent.set_ylim(0, game.size * game.size * 0.15)
    ax_recent.set_xlabel('Generation', fontsize=9)
    ax_recent.set_ylabel('Population', fontsize=9)
    ax_recent.grid(True, alpha=0.3)
    ax_recent.set_title('Recent Dynamics (Last 100 Gen)', fontsize=10)

    # Right panel: Species legend and statistics
    ax_legend = fig.add_subplot(gs[:, 2])
    ax_legend.axis('off')
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)

    # Background box
    from matplotlib.patches import Rectangle
    bg_box = Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax_legend.transAxes,
                       facecolor='white', edgecolor='gray', alpha=0.95, zorder=0)
    ax_legend.add_patch(bg_box)

    # Store legend elements for updating
    legend_patches = []
    legend_texts = []

    frame_counter = [0]  # Use list for nonlocal mutation

    def update(_frame):
        nonlocal recent_species_lines, global_species_lines

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
        while len(recent_species_lines) < len(SPECIES_CONFIG):
            sp_id = len(recent_species_lines)
            color = SPECIES_CONFIG[sp_id]['color']
            line1, = ax_recent.plot([], [], color=color, linewidth=1.5)
            recent_species_lines.append(line1)
            line2, = ax_global.plot([], [], color=color, linewidth=1.5)
            global_species_lines.append(line2)

        # Always update display (fast operation)
        img_display.set_array(game.render())
        ax_main.set_title(f'Environment - Gen {game.generation}', fontsize=10)

        # Update charts only every CHART_UPDATE_INTERVAL frames (expensive operations)
        if should_update_charts:
            history_len = len(game.history['population'])
            if HISTORY_WINDOW > 0 and history_len > 0:
                # With sliding window, x starts from (current_gen - history_len + 1)
                start_gen = game.generation - history_len + 1
                gens = list(range(start_gen, game.generation + 1))
            else:
                gens = list(range(history_len))

            # Update global chart (full history)
            for sp_id in range(len(global_species_lines)):
                if sp_id < len(game.history['species']):
                    is_extinct = SPECIES_CONFIG[sp_id].get('extinct', False)
                    global_species_lines[sp_id].set_visible(not is_extinct)
                    global_species_lines[sp_id].set_data(gens, game.history['species'][sp_id])
            line_global_total.set_data(gens, game.history['population'])

            # Update recent chart (last 100 generations)
            recent_window = min(100, history_len)
            recent_gens = gens[-recent_window:] if gens else []
            for sp_id in range(len(recent_species_lines)):
                if sp_id < len(game.history['species']):
                    is_extinct = SPECIES_CONFIG[sp_id].get('extinct', False)
                    recent_species_lines[sp_id].set_visible(not is_extinct)
                    recent_data = game.history['species'][sp_id][-recent_window:]
                    recent_species_lines[sp_id].set_data(recent_gens, recent_data)

            if recent_gens:
                line_recent_total.set_data(recent_gens, game.history['population'][-recent_window:])
                ax_recent.set_xlim(recent_gens[0], recent_gens[-1])

            # Update x/y limits
            ax_global.set_xlim(gens[0] if gens else 0, max(gens[-1] if gens else 100, 100))

            if game.history['population']:
                max_pop = max(game.history['population']) * 1.2
                ax_recent.set_ylim(0, max(max_pop, 100))
                ax_global.set_ylim(0, max(max_pop, 100))

        # Update species legend panel (right side)
        total = game.history['population'][-1] if game.history['population'] else 0
        alive_species_count = sum(1 for sp in SPECIES_CONFIG if not sp.get('extinct', False))

        # Clear previous legend elements
        for patch in legend_patches:
            patch.remove()
        for text in legend_texts:
            text.remove()
        legend_patches.clear()
        legend_texts.clear()

        # Header information
        y_pos = 0.96
        header_lines = [
            f"Gen: {game.generation}",
            f"Species: {alive_species_count}",
            f"Population: {total}",
            f"Step: {elapsed*1000:.1f}ms",
            "=" * 30
        ]
        for line in header_lines:
            txt = ax_legend.text(0.05, y_pos, line, transform=ax_legend.transAxes,
                                fontsize=8, family='monospace', va='top')
            legend_texts.append(txt)
            y_pos -= 0.03

        y_pos -= 0.01  # Extra space after header

        # Build species list with color, number, count, and percentage
        species_data = []
        for sp_id in range(len(SPECIES_CONFIG)):
            if SPECIES_CONFIG[sp_id].get('extinct', False):
                continue  # Skip extinct species

            if sp_id < len(game.history['species']) and game.history['species'][sp_id]:
                count = game.history['species'][sp_id][-1]
            else:
                count = 0

            # Mark species as extinct if count is 0
            if count == 0:
                if not SPECIES_CONFIG[sp_id]['extinct']:
                    SPECIES_CONFIG[sp_id]['extinct'] = True
                    SPECIES_CONFIG[sp_id]['extinct_gen'] = game.generation
                    # Clear blob separation tracker for extinct species
                    if sp_id in game.blob_separation_tracker:
                        del game.blob_separation_tracker[sp_id]
                continue

            name = SPECIES_CONFIG[sp_id]['name']
            pct = (count / total * 100) if total > 0 else 0
            color = SPECIES_CONFIG[sp_id]['color']
            species_data.append((sp_id, name, count, pct, color))

        # Sort by population (descending)
        species_data.sort(key=lambda x: x[2], reverse=True)

        # Draw species info with colored squares
        from matplotlib.patches import Rectangle
        for sp_id, name, count, pct, color in species_data:
            if y_pos < 0.05:  # Stop if we run out of space
                break

            # Draw colored square
            square = Rectangle((0.05, y_pos - 0.015), 0.02, 0.02,
                              transform=ax_legend.transAxes,
                              facecolor=color, edgecolor='black', linewidth=0.5)
            ax_legend.add_patch(square)
            legend_patches.append(square)

            # Draw text: name, count, percentage
            txt = ax_legend.text(0.08, y_pos, f"{name:10s} {count:6d} ({pct:5.1f}%)",
                                transform=ax_legend.transAxes,
                                fontsize=7, family='monospace', va='top')
            legend_texts.append(txt)
            y_pos -= 0.025

        # Extinct message
        if game.is_extinct():
            y_pos -= 0.02
            extinct_lines = ["=" * 30, "    EXTINCT!", "=" * 30]
            for line in extinct_lines:
                txt = ax_legend.text(0.05, y_pos, line, transform=ax_legend.transAxes,
                                    fontsize=9, family='monospace', va='top',
                                    color='red', weight='bold')
                legend_texts.append(txt)
                y_pos -= 0.03

        return [img_display, *recent_species_lines, line_recent_total,
                *global_species_lines, line_global_total, *legend_patches, *legend_texts]

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
