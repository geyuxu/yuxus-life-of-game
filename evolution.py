"""
Yuxu's Game of Life - A GPU-accelerated Genome-Based Evolution Simulation

A neuroevolution ecosystem where organisms compete, reproduce, and evolve through:
- Genome-Based Identity: 12-dimensional genome determines organism identity and compatibility
- Neuroevolution: Neural network weights inherited and mutated during reproduction
- Reinforcement Learning: Continuous learning through policy gradient updates
- Lineage Tracking: Validates training effectiveness by tracking trained vs random lineages
- GPU Acceleration: All computations run on GPU (CUDA/MPS) when available
"""

import torch
import torch.nn.functional as F
import numpy as np
import colorsys
import time
import os
import argparse
from scipy import ndimage
import threading
import queue

# Import configuration
from config import *

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
# NOTE: All simulation parameters are now imported from config.py
# =============================================================================


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

        # Time-based auto-save tracking
        import time
        self.last_save_time = time.time()

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

        # Validation tracking: mark cells with trained weights vs random
        # Trained generation tracking: -1=random, 0=direct inheritance, 1,2,3...=generation from trained
        self.trained_generation = torch.full((size, size), -1, dtype=torch.int32, device=DEVICE)
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

        # Pre-allocated buffers for _fill_enclosed_spaces() to reduce memory allocation overhead
        self._buffer_alive_neighbor_count = torch.zeros((size, size), dtype=torch.int32, device=DEVICE)
        self._buffer_reference_genome = torch.zeros((size, size, 12), dtype=torch.float32, device=DEVICE)
        self._buffer_reference_found = torch.zeros((size, size), dtype=torch.bool, device=DEVICE)
        self._buffer_similar_neighbor_count = torch.zeros((size, size), dtype=torch.int32, device=DEVICE)
        self._buffer_neighbor_genomes_sum = torch.zeros((size, size, 12), dtype=torch.float32, device=DEVICE)
        self._buffer_neighbor_w1_sum = torch.zeros((size, size, INPUT_SIZE, MAX_HIDDEN_SIZE), dtype=torch.float32, device=DEVICE)
        self._buffer_neighbor_w2_sum = torch.zeros((size, size, MAX_HIDDEN_SIZE, NUM_ACTIONS), dtype=torch.float32, device=DEVICE)
        self._buffer_neighbor_trained_gen_sum = torch.zeros((size, size), dtype=torch.int32, device=DEVICE)
        self._buffer_neighbor_trained_gen_count = torch.zeros((size, size), dtype=torch.int32, device=DEVICE)

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

                # Mark these cells as generation 0 (direct inheritance from trained weights)
                self.trained_generation[elite_rows, elite_cols] = 0

                print(f"{num_elite} cells inherited saved weights (50% saved + 50% random for diversity)")
                print(f"Generational tracking enabled: Gen0(trained) vs descendants vs random")

        # Update genome neural fingerprints based on initialized weights
        for i in range(num_cells):
            r, c = rows[i].item(), cols[i].item()
            genome_np = self.compute_genome(r, c)
            if genome_np is not None:
                self.genome[r, c] = torch.from_numpy(genome_np).float().to(DEVICE)

                # Add strong random variation to neural fingerprint for diversity
                # This prevents all cells from having similar genomes even with 50% same weights
                diversity_boost = torch.randn(8, device=DEVICE) * 1.5
                self.genome[r, c, 0:8] += diversity_boost

                # Also add variation to chemical affinity based on position
                pos_seed = (r * self.size + c) * 0.01
                chem_variation = torch.randn(NUM_CHEMICALS, device=DEVICE) * 1.0 + pos_seed
                self.genome[r, c, 8:12] += chem_variation

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

        # Fill enclosed spaces with new cells (only every 10 steps to keep system dynamic)
        if self.generation % 10 == 0:
            self._fill_enclosed_spaces()

        # Tissue fission: split large enclosed tissues (every 50 steps)
        if self.generation % 50 == 0:
            self._tissue_fission()

        # Dominance check: force mutation on dominant species (every N generations)
        if self.generation > 0 and self.generation % DOMINANCE_CHECK_INTERVAL == 0:
            self._check_dominance_and_mutate()

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

        # Save best network periodically (time-based auto-save)
        self._update_best_network()
        if AUTO_SAVE_ENABLED and self.generation > 0:
            import time
            current_time = time.time()
            elapsed = current_time - self.last_save_time
            if elapsed >= SAVE_INTERVAL_SECONDS:
                self._save_best_weights()
                self.last_save_time = current_time
                print(f"\n[AUTO-SAVE] Generation {self.generation}, Population {self.alive.sum().item()}")

        # Print validation report every 200 generations
        if self.generation > 0 and self.generation % 200 == 0:
            self.print_validation_report()

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

            # Check if target position is "enclosed" by similar genomes (safer for offspring)
            # Count how many of the 8 neighbors around target are alive with similar genome to parent
            target_enclosure = torch.zeros((self.size, self.size), dtype=torch.float32, device=DEVICE)

            # Get parent genomes (cells attempting to reproduce at self.rows, self.cols)
            parent_genomes = self.genome[self.rows, self.cols]  # [H, W, 12]

            for check_dr, check_dc in dirs:
                check_r = (target_r + check_dr) % self.size
                check_c = (target_c + check_dc) % self.size
                neighbor_alive = self.alive[check_r, check_c]

                # Get genome of the target's neighbors
                neighbor_genome = self.genome[check_r, check_c]  # [H, W, 12]

                # Calculate genome distance between neighbor and parent
                genome_dist = torch.norm(neighbor_genome - parent_genomes, dim=-1)  # [H, W]
                is_similar = (genome_dist < MATE_GENOME_THRESHOLD).float()

                # Add to enclosure count where neighbor is alive and similar
                target_enclosure += neighbor_alive.float() * is_similar

            # Enclosed spaces (≥6 similar neighbors) get reproduction bonus
            # This encourages forming protective structures
            is_enclosed = target_enclosure >= 6
            reproduction_threshold = torch.where(is_enclosed, 0.3, 0.5)  # LOWER threshold = easier to pass

            can_reproduce = is_reproduce & target_empty & (self.energy >= SPECIES_REPRO_THRESHOLD)
            winner = can_reproduce & (rand_shared > reproduction_threshold)  # rand > 0.3 = 70% chance (enclosed)
                                                                                # rand > 0.5 = 50% chance (open)

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

            # Inherit trained generation from parents
            parent1_gen = self.trained_generation[parent1_r, parent1_c]
            parent2_gen = self.trained_generation[mate_r, mate_c]

            # If either parent has trained ancestry, child is next generation
            # Use the closer ancestor (smaller generation number)
            has_trained_parent = (parent1_gen >= 0) | (parent2_gen >= 0)
            if has_trained_parent.any():
                # Filter out -1 values, take minimum of valid generations, then add 1
                valid_gens = torch.stack([parent1_gen, parent2_gen], dim=0)
                valid_gens = torch.where(valid_gens >= 0, valid_gens, torch.tensor(999, device=DEVICE))
                min_gen = valid_gens.min(dim=0)[0]
                child_gen = torch.where(min_gen < 999, min_gen + 1, torch.tensor(-1, dtype=torch.int32, device=DEVICE))
                self.trained_generation[child_r, child_c] = child_gen
            else:
                # Both parents are random (-1), child is also random
                self.trained_generation[child_r, child_c] = -1

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

    def _fill_enclosed_spaces(self):
        """
        Fill enclosed spaces with new cells of the surrounding species.

        If an empty cell has at least 6 alive neighbors with similar genomes,
        create a new cell that inherits from the similar neighbors.
        """
        # Find all empty cells
        empty_mask = ~self.alive

        if not empty_mask.any():
            return

        # For each empty cell, check all 8 neighbors
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

        # Use pre-allocated buffers and zero them out
        self._buffer_alive_neighbor_count.zero_()
        self._buffer_reference_genome.zero_()
        self._buffer_reference_found.zero_()

        # Alias for readability
        alive_neighbor_count = self._buffer_alive_neighbor_count
        reference_genome = self._buffer_reference_genome
        reference_found = self._buffer_reference_found

        for dr, dc in dirs:
            neighbor_r = (self.rows + dr) % self.size
            neighbor_c = (self.cols + dc) % self.size
            neighbor_alive = self.alive[neighbor_r, neighbor_c]
            neighbor_genome = self.genome[neighbor_r, neighbor_c]

            # Count alive neighbors
            alive_neighbor_count += neighbor_alive.int()

            # Set first alive neighbor as reference genome
            need_reference = ~reference_found & neighbor_alive
            reference_genome = torch.where(
                need_reference.unsqueeze(-1),
                neighbor_genome,
                reference_genome
            )
            reference_found |= neighbor_alive

        # Second pass: use pre-allocated buffers and zero them out
        self._buffer_similar_neighbor_count.zero_()
        self._buffer_neighbor_genomes_sum.zero_()
        self._buffer_neighbor_w1_sum.zero_()
        self._buffer_neighbor_w2_sum.zero_()
        self._buffer_neighbor_trained_gen_sum.zero_()
        self._buffer_neighbor_trained_gen_count.zero_()

        # Alias for readability
        similar_neighbor_count = self._buffer_similar_neighbor_count
        neighbor_genomes_sum = self._buffer_neighbor_genomes_sum
        neighbor_w1_sum = self._buffer_neighbor_w1_sum
        neighbor_w2_sum = self._buffer_neighbor_w2_sum
        neighbor_trained_gen_sum = self._buffer_neighbor_trained_gen_sum
        neighbor_trained_gen_count = self._buffer_neighbor_trained_gen_count

        for dr, dc in dirs:
            neighbor_r = (self.rows + dr) % self.size
            neighbor_c = (self.cols + dc) % self.size
            neighbor_alive = self.alive[neighbor_r, neighbor_c]
            neighbor_genome = self.genome[neighbor_r, neighbor_c]

            # Check genome similarity to reference (only where neighbor is alive)
            genome_dist = torch.norm(neighbor_genome - reference_genome, dim=-1)
            is_similar = neighbor_alive & (genome_dist < MATE_GENOME_THRESHOLD)

            # Count similar neighbors
            similar_neighbor_count += is_similar.int()

            # Accumulate data from similar neighbors
            is_similar_3d = is_similar.unsqueeze(-1)
            neighbor_genomes_sum += torch.where(is_similar_3d, neighbor_genome, torch.zeros_like(neighbor_genome))

            is_similar_w1 = is_similar.unsqueeze(-1).unsqueeze(-1)
            neighbor_w1_sum += torch.where(is_similar_w1, self.w1[neighbor_r, neighbor_c], torch.zeros_like(self.w1[neighbor_r, neighbor_c]))

            is_similar_w2 = is_similar.unsqueeze(-1).unsqueeze(-1)
            neighbor_w2_sum += torch.where(is_similar_w2, self.w2[neighbor_r, neighbor_c], torch.zeros_like(self.w2[neighbor_r, neighbor_c]))

            # Accumulate trained_generation from similar neighbors (only if >= 0)
            neighbor_gen = self.trained_generation[neighbor_r, neighbor_c]
            has_trained_gen = is_similar & (neighbor_gen >= 0)
            neighbor_trained_gen_sum += torch.where(has_trained_gen, neighbor_gen, torch.zeros_like(neighbor_gen))
            neighbor_trained_gen_count += has_trained_gen.int()

        # Fill cells that are COMPLETELY enclosed (all 8 neighbors alive and similar)
        # This prevents over-filling and keeps the system dynamic
        fill_mask = empty_mask & (similar_neighbor_count >= 8)

        if not fill_mask.any():
            return

        # Vectorized batch operations for all fill positions at once
        # Avoid division by zero: where count==0, use 1 (these positions won't be filled anyway)
        safe_count = torch.where(similar_neighbor_count > 0, similar_neighbor_count, torch.ones_like(similar_neighbor_count))

        # Calculate averages for all positions (3D operations)
        avg_genome = neighbor_genomes_sum / safe_count.unsqueeze(-1).float()
        avg_w1 = neighbor_w1_sum / safe_count.unsqueeze(-1).unsqueeze(-1).float()
        avg_w2 = neighbor_w2_sum / safe_count.unsqueeze(-1).unsqueeze(-1).float()

        # Calculate trained_generation for all positions
        # Where neighbor_trained_gen_count > 0, calculate avg and add 1; otherwise -1
        safe_trained_count = torch.where(neighbor_trained_gen_count > 0, neighbor_trained_gen_count, torch.ones_like(neighbor_trained_gen_count))
        avg_trained_gen = neighbor_trained_gen_sum.float() / safe_trained_count.float()
        child_gen = torch.where(neighbor_trained_gen_count > 0, avg_trained_gen.int() + 1, torch.full_like(neighbor_trained_gen_count, -1))

        # Generate mutation noise for all cells at once
        mutation_noise_genome = torch.randn_like(self.genome) * MUTATION_RATE * 0.5
        mutation_noise_w1 = torch.randn_like(self.w1) * MUTATION_RATE * 0.5
        mutation_noise_w2 = torch.randn_like(self.w2) * MUTATION_RATE * 0.5

        # Apply mutations and create new cells (batch assignment using fill_mask)
        self.genome = torch.where(fill_mask.unsqueeze(-1), avg_genome + mutation_noise_genome, self.genome)
        self.w1 = torch.where(fill_mask.unsqueeze(-1).unsqueeze(-1), avg_w1 + mutation_noise_w1, self.w1)
        self.w2 = torch.where(fill_mask.unsqueeze(-1).unsqueeze(-1), avg_w2 + mutation_noise_w2, self.w2)

        # Update all state tensors in batch
        self.alive[fill_mask] = True
        self.energy[fill_mask] = SPECIES_OFFSPRING_ENERGY
        self.is_newborn[fill_mask] = True
        self.lifetime[fill_mask] = 0
        self.repro_count[fill_mask] = 0
        self.hunger[fill_mask] = 0
        self.trained_generation[fill_mask] = child_gen[fill_mask]

    def _tissue_fission(self):
        """
        Split enclosed tissue into two organisms with genetic recombination.

        Each organism retains half neurons (interleaved) and fills the other half randomly.
        This simulates asexual reproduction at the organism level.
        """
        alive = self.alive.cpu().numpy()
        genome = self.genome.cpu().numpy()

        if not alive.any():
            return

        # Find connected tissue regions
        visited = np.zeros_like(alive, dtype=bool)
        tissues = []

        for y in range(self.size):
            for x in range(self.size):
                if not alive[y, x] or visited[y, x]:
                    continue

                # BFS to find connected tissue
                tissue_cells = []
                queue = [(y, x)]
                visited[y, x] = True
                ref_genome = genome[y, x]

                while queue:
                    cy, cx = queue.pop(0)
                    tissue_cells.append((cy, cx))

                    # Check 4 neighbors
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny = (cy + dy) % self.size
                        nx = (cx + dx) % self.size

                        if not visited[ny, nx] and alive[ny, nx]:
                            neighbor_genome = genome[ny, nx]
                            dist = np.linalg.norm(neighbor_genome - ref_genome)
                            if dist < MATE_GENOME_THRESHOLD:
                                visited[ny, nx] = True
                                queue.append((ny, nx))

                # Only consider tissues large enough to split (>= 100 cells)
                if len(tissue_cells) >= 100:
                    tissues.append(tissue_cells)

        if not tissues:
            return

        # Process each tissue for potential fission
        for tissue_cells in tissues:
            # 10% chance of fission per generation
            if np.random.random() > 0.1:
                continue

            # Find tissue center
            coords = np.array(tissue_cells)
            center_y = coords[:, 0].mean()
            center_x = coords[:, 1].mean()

            # Divide tissue into two halves based on distance from center
            # Use a random division axis
            angle = np.random.random() * np.pi
            division_vec = np.array([np.cos(angle), np.sin(angle)])

            half_a = []
            half_b = []

            for y, x in tissue_cells:
                # Project cell position onto division vector
                pos_vec = np.array([y - center_y, x - center_x])
                projection = np.dot(pos_vec, division_vec)

                if projection >= 0:
                    half_a.append((y, x))
                else:
                    half_b.append((y, x))

            # Skip if division is too unbalanced
            if len(half_a) < 20 or len(half_b) < 20:
                continue

            # Apply genetic recombination to each half
            self._apply_fission_mutation(half_a)
            self._apply_fission_mutation(half_b)

    def _apply_random_mutation(self, cells, random_ratio=0.7, energy_cost=False):
        """
        Apply random mutation to a group of cells.

        Args:
            cells: List of (y, x) cell positions
            random_ratio: Ratio of neurons to randomize (0.3 to 0.7)
                         0.3 = 30% randomize, 70% keep (mild)
                         0.5 = 50% randomize, 50% keep (moderate)
                         0.7 = 70% randomize, 30% keep (strong, default)
            energy_cost: If True, reduce energy by 30% (for fission cost)
        """
        # Calculate how many neurons to keep (inverse of random_ratio)
        keep_ratio = 1.0 - random_ratio
        neuron_mask = torch.zeros(MAX_HIDDEN_SIZE, dtype=torch.bool, device=DEVICE)
        num_keep = int(MAX_HIDDEN_SIZE * keep_ratio)
        keep_indices = torch.randperm(MAX_HIDDEN_SIZE, device=DEVICE)[:num_keep]
        neuron_mask[keep_indices] = True

        for y, x in cells:
            # Get current weights
            w1_current = self.w1[y, x].clone()
            w2_current = self.w2[y, x].clone()
            genome_current = self.genome[y, x].clone()

            # Create new weights: randomly keep some neurons, randomize others
            w1_new = torch.randn_like(w1_current) * 0.5
            w2_new = torch.randn_like(w2_current) * 0.5

            # Apply mask: keep selected neurons, randomize others
            # w1: [INPUT_SIZE, MAX_HIDDEN_SIZE]
            w1_new[:, neuron_mask] = w1_current[:, neuron_mask]

            # w2: [MAX_HIDDEN_SIZE, NUM_ACTIONS]
            w2_new[neuron_mask, :] = w2_current[neuron_mask, :]

            # Update weights
            self.w1[y, x] = w1_new
            self.w2[y, x] = w2_new

            # Update genome with mutation
            genome_mutation = torch.randn(12, device=DEVICE) * MUTATION_RATE
            self.genome[y, x] = genome_current + genome_mutation

            # Optional energy cost (for fission)
            if energy_cost:
                self.energy[y, x] = max(5.0, self.energy[y, x].item() * 0.7)

    def _apply_fission_mutation(self, cells):
        """
        Apply genetic recombination for tissue fission.
        Uses 50% randomization (moderate) with energy cost.
        """
        self._apply_random_mutation(cells, random_ratio=0.5, energy_cost=True)

    def _check_dominance_and_mutate(self):
        """
        Check for dominant species and force mutation to maintain diversity.

        If a genome cluster exceeds DOMINANCE_THRESHOLD (30%) of population,
        apply interleaved mutation: keep 50% neurons, randomize other 50%.
        """
        alive = self.alive.cpu().numpy()
        genome = self.genome.cpu().numpy()

        if not alive.any():
            return

        total_population = alive.sum()
        alive_genomes = genome[alive]
        N = len(alive_genomes)

        if N < 100:  # Skip if population too small
            return

        # Sample for clustering (for performance)
        max_sample = 500
        if N > max_sample:
            sample_indices = np.random.choice(N, max_sample, replace=False)
            sample_genomes = alive_genomes[sample_indices]
            N_sample = max_sample
        else:
            sample_genomes = alive_genomes
            N_sample = N

        # Build adjacency matrix
        adjacency = np.zeros((N_sample, N_sample), dtype=bool)
        for i in range(N_sample):
            for j in range(i+1, N_sample):
                dist = np.linalg.norm(sample_genomes[i] - sample_genomes[j])
                if dist < MATE_GENOME_THRESHOLD:
                    adjacency[i, j] = True
                    adjacency[j, i] = True

        # Find cluster centers using BFS
        cluster_centers = []
        visited = np.zeros(N_sample, dtype=bool)

        for i in range(N_sample):
            if visited[i]:
                continue

            cluster_indices = []
            queue = [i]
            visited[i] = True

            while queue:
                current = queue.pop(0)
                cluster_indices.append(current)

                neighbors = np.where(adjacency[current] & ~visited)[0]
                for neighbor in neighbors:
                    visited[neighbor] = True
                    queue.append(neighbor)

            if cluster_indices:
                cluster_genomes = sample_genomes[cluster_indices]
                center = cluster_genomes.mean(axis=0)
                cluster_centers.append(center)

        if len(cluster_centers) == 0:
            return

        # Assign all alive cells to nearest cluster and count
        cluster_centers_array = np.array(cluster_centers)
        cluster_counts = [0] * len(cluster_centers)
        cluster_cell_positions = [[] for _ in range(len(cluster_centers))]

        positions = np.argwhere(alive)
        for y, x in positions:
            cell_genome = genome[y, x]
            distances = np.linalg.norm(cluster_centers_array - cell_genome, axis=1)
            nearest_cluster = np.argmin(distances)
            cluster_counts[nearest_cluster] += 1
            cluster_cell_positions[nearest_cluster].append((y, x))

        # Check for dominant clusters and apply mutation
        for cluster_id, count in enumerate(cluster_counts):
            ratio = count / total_population

            if ratio > DOMINANCE_THRESHOLD:
                print(f"\n[DOMINANCE MUTATION] Cluster {cluster_id} = {ratio*100:.1f}% (>{DOMINANCE_THRESHOLD*100:.0f}%)")
                print(f"  Applying interleaved mutation to {count} cells...")

                # Apply interleaved mutation to all cells in this cluster
                cells = cluster_cell_positions[cluster_id]
                self._apply_interleaved_mutation(cells)

    def _apply_interleaved_mutation(self, cells):
        """
        Apply strong mutation for dominance control.
        Uses 70% randomization (strong) without energy cost.
        """
        self._apply_random_mutation(cells, random_ratio=0.7, energy_cost=False)

    def get_validation_stats(self):
        """
        Compare performance by trained generation:
        - Gen0: Direct inheritance from trained weights
        - Gen1+: Descendants of trained cells
        - Random: No trained ancestry
        """
        # Group by generation
        gen0_mask = self.alive & (self.trained_generation == 0)
        gen1_5_mask = self.alive & (self.trained_generation >= 1) & (self.trained_generation <= 5)
        gen6plus_mask = self.alive & (self.trained_generation > 5)
        random_mask = self.alive & (self.trained_generation < 0)

        gen0_count = gen0_mask.sum().item()
        gen1_5_count = gen1_5_mask.sum().item()
        gen6plus_count = gen6plus_mask.sum().item()
        random_count = random_mask.sum().item()

        total = gen0_count + gen1_5_count + gen6plus_count + random_count
        if total == 0:
            return None

        stats = {
            'gen0_count': gen0_count,
            'gen1_5_count': gen1_5_count,
            'gen6plus_count': gen6plus_count,
            'random_count': random_count,
            'trained_total': gen0_count + gen1_5_count + gen6plus_count,
        }

        # Gen 0 stats
        if gen0_count > 0:
            stats['gen0_avg_lifetime'] = self.lifetime[gen0_mask].float().mean().item()
            stats['gen0_avg_repro'] = self.repro_count[gen0_mask].float().mean().item()
            stats['gen0_avg_energy'] = self.energy[gen0_mask].float().mean().item()
        else:
            stats['gen0_avg_lifetime'] = 0
            stats['gen0_avg_repro'] = 0
            stats['gen0_avg_energy'] = 0

        # Gen 1-5 stats
        if gen1_5_count > 0:
            stats['gen1_5_avg_lifetime'] = self.lifetime[gen1_5_mask].float().mean().item()
            stats['gen1_5_avg_repro'] = self.repro_count[gen1_5_mask].float().mean().item()
            stats['gen1_5_avg_energy'] = self.energy[gen1_5_mask].float().mean().item()
        else:
            stats['gen1_5_avg_lifetime'] = 0
            stats['gen1_5_avg_repro'] = 0
            stats['gen1_5_avg_energy'] = 0

        # Gen 6+ stats
        if gen6plus_count > 0:
            stats['gen6plus_avg_lifetime'] = self.lifetime[gen6plus_mask].float().mean().item()
            stats['gen6plus_avg_repro'] = self.repro_count[gen6plus_mask].float().mean().item()
            stats['gen6plus_avg_energy'] = self.energy[gen6plus_mask].float().mean().item()
        else:
            stats['gen6plus_avg_lifetime'] = 0
            stats['gen6plus_avg_repro'] = 0
            stats['gen6plus_avg_energy'] = 0

        # Random stats
        if random_count > 0:
            stats['random_avg_lifetime'] = self.lifetime[random_mask].float().mean().item()
            stats['random_avg_repro'] = self.repro_count[random_mask].float().mean().item()
            stats['random_avg_energy'] = self.energy[random_mask].float().mean().item()
        else:
            stats['random_avg_lifetime'] = 0
            stats['random_avg_repro'] = 0
            stats['random_avg_energy'] = 0

        # Combined trained lineage stats (Gen0 + Gen1-5 + Gen6+)
        trained_lineage_mask = self.alive & (self.trained_generation >= 0)
        trained_lineage_count = trained_lineage_mask.sum().item()
        if trained_lineage_count > 0:
            stats['trained_lineage_avg_lifetime'] = self.lifetime[trained_lineage_mask].float().mean().item()
            stats['trained_lineage_avg_repro'] = self.repro_count[trained_lineage_mask].float().mean().item()
            stats['trained_lineage_avg_energy'] = self.energy[trained_lineage_mask].float().mean().item()
        else:
            stats['trained_lineage_avg_lifetime'] = 0
            stats['trained_lineage_avg_repro'] = 0
            stats['trained_lineage_avg_energy'] = 0

        return stats

    def print_validation_report(self):
        """Print generational validation report."""
        stats = self.get_validation_stats()
        if stats is None:
            return

        print("\n" + "="*70)
        print(f"GENERATIONAL VALIDATION - Generation {self.generation}")
        print("="*70)

        total = stats['gen0_count'] + stats['gen1_5_count'] + stats['gen6plus_count'] + stats['random_count']

        print(f"\nPopulation by Lineage:")
        print(f"  Gen 0 (Direct trained):       {stats['gen0_count']:5d} ({stats['gen0_count']/total*100:5.1f}%)")
        print(f"  Gen 1-5 (Near descendants):   {stats['gen1_5_count']:5d} ({stats['gen1_5_count']/total*100:5.1f}%)")
        print(f"  Gen 6+ (Far descendants):     {stats['gen6plus_count']:5d} ({stats['gen6plus_count']/total*100:5.1f}%)")
        print(f"  Random (No trained ancestry): {stats['random_count']:5d} ({stats['random_count']/total*100:5.1f}%)")
        print(f"  ----------------------------------------")
        print(f"  Total Trained Lineage:        {stats['trained_total']:5d} ({stats['trained_total']/total*100:5.1f}%)")

        print(f"\n** PRIMARY COMPARISON: Trained Lineage vs Random **")
        print(f"  Lifetime:     {stats['trained_lineage_avg_lifetime']:6.1f} vs {stats['random_avg_lifetime']:6.1f}", end="")
        if stats['random_avg_lifetime'] > 0 and stats['trained_lineage_avg_lifetime'] > 0:
            ratio = stats['trained_lineage_avg_lifetime'] / stats['random_avg_lifetime']
            print(f"  = {ratio:5.2f}x {'✓ BETTER' if ratio > 1 else '✗ WORSE'}")
        else:
            print()

        print(f"  Reproduction: {stats['trained_lineage_avg_repro']:6.2f} vs {stats['random_avg_repro']:6.2f}", end="")
        if stats['random_avg_repro'] > 0 and stats['trained_lineage_avg_repro'] > 0:
            ratio = stats['trained_lineage_avg_repro'] / stats['random_avg_repro']
            print(f"  = {ratio:5.2f}x {'✓ BETTER' if ratio > 1 else '✗ WORSE'}")
        else:
            print()

        print(f"  Energy:       {stats['trained_lineage_avg_energy']:6.1f} vs {stats['random_avg_energy']:6.1f}", end="")
        if stats['random_avg_energy'] > 0 and stats['trained_lineage_avg_energy'] > 0:
            ratio = stats['trained_lineage_avg_energy'] / stats['random_avg_energy']
            print(f"  = {ratio:5.2f}x {'✓ BETTER' if ratio > 1 else '✗ WORSE'}")
        else:
            print()

        print(f"\nBreakdown by Generation:")
        print(f"  Lifetime:     Gen0={stats['gen0_avg_lifetime']:5.1f}  Gen1-5={stats['gen1_5_avg_lifetime']:5.1f}  Gen6+={stats['gen6plus_avg_lifetime']:5.1f}")
        print(f"  Reproduction: Gen0={stats['gen0_avg_repro']:5.2f}  Gen1-5={stats['gen1_5_avg_repro']:5.2f}  Gen6+={stats['gen6plus_avg_repro']:5.2f}")
        print(f"  Energy:       Gen0={stats['gen0_avg_energy']:5.1f}  Gen1-5={stats['gen1_5_avg_energy']:5.1f}  Gen6+={stats['gen6plus_avg_energy']:5.1f}")

        print("="*70 + "\n")

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
    print(f"\n[Dominance-Based Mutation]")
    print(f"  Force mutation when species exceeds {DOMINANCE_THRESHOLD*100:.0f}% of population")
    print(f"  Check every {DOMINANCE_CHECK_INTERVAL} generations")
    print(f"  Mutation: Keep 30% neurons (random), randomize 70%")
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


