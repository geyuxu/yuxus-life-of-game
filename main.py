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
GRID_SIZE = 300 #ARGS.grid
INITIAL_ENERGY = 30.0
MAX_ENERGY = 100.0
MOVE_COST = 0.2
CROWDING_THRESHOLD = 4
CROWDING_PENALTY = 0.5

# Species configuration
INITIAL_NUM_SPECIES = 5
MAX_SPECIES = 500
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
BLOB_SEPARATION_DELAY = 0   # Generations of separation before blob becomes new species (0 = immediate)
BLOB_CHECK_INTERVAL = 100   # Check for blob separation every N generations

# Performance tuning
HISTORY_WINDOW = 1000       # Keep only last N generations in history (0 = unlimited)
RENDER_INTERVAL = 1         # Render every N frames (increase for faster simulation)
CHART_UPDATE_INTERVAL = 5   # Update charts every N frames (reduces matplotlib overhead)

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
GENOME_COLOR_UPDATE_INTERVAL = 50  # Update species colors every N generations

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
# SPECIES MANAGEMENT
# =============================================================================
SPECIES_CONFIG = []
SPECIES_CHILD_COUNT = {}


def generate_random_color(index: int) -> tuple:
    """Generate a visually distinct color using HSV color space with golden ratio.

    Args:
        index: Species ID (used to generate unique color)
    """
    # Use golden ratio to maximize color difference between adjacent species
    # This ensures new species have visually distinct colors even with many species
    GOLDEN_RATIO = 0.618033988749895
    hue = (index * GOLDEN_RATIO) % 1.0

    # Vary saturation and value for additional distinction
    # Use index-based seed for reproducibility
    np.random.seed(index + 1000)
    saturation = 0.65 + np.random.random() * 0.35
    value = 0.70 + np.random.random() * 0.30

    return colorsys.hsv_to_rgb(hue, saturation, value)


def create_species_config(sp_id: int, num_total: int, name: str = None) -> dict:
    """Create configuration for a single species."""
    prey_list = [j for j in range(num_total) if j != sp_id]

    # Initialize random chemical preferences (evolvable)
    # Each species has affinity to different chemicals
    np.random.seed(sp_id + 5000)  # Different seed for chemical preferences
    chemical_affinity = np.random.randn(NUM_CHEMICALS) * 0.5

    return {
        'name': name or f'S{sp_id}',
        'color': generate_random_color(sp_id),
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
        'chemical_affinity': chemical_affinity.tolist(),  # Preference for each chemical type
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

        # Inherit and mutate chemical affinity from parent
        parent_chem = np.array(SPECIES_CONFIG[parent_id]['chemical_affinity'])
        child_chem = parent_chem + np.random.randn(NUM_CHEMICALS) * SPLIT_MUTATION_RATE

        SPECIES_CONFIG[new_id] = {
            'name': new_name,
            'color': generate_random_color(new_id),
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
            'chemical_affinity': child_chem.tolist(),
        }

        # Reset child count for recycled slot
        SPECIES_CHILD_COUNT[new_id] = 0

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

        # Inherit and mutate chemical affinity from parent
        parent_chem = np.array(SPECIES_CONFIG[parent_id]['chemical_affinity'])
        child_chem = parent_chem + np.random.randn(NUM_CHEMICALS) * SPLIT_MUTATION_RATE
        new_config['chemical_affinity'] = child_chem.tolist()

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

    Uses first 3 genome dimensions (w1_mean, w1_std, w1_abs_mean) to generate:
    - Hue: Normalized w1_mean (range 0-1)
    - Saturation: Normalized w1_std (range 0.5-1.0)
    - Value: Normalized w1_abs_mean (range 0.6-1.0)

    Args:
        genome: 12-dimensional genome vector

    Returns:
        RGB tuple (3 floats in range 0-1)
    """
    # Use first 3 neural features for color mapping
    w1_mean = genome[0]
    w1_std = genome[1]
    w1_abs_mean = genome[2]

    # Map to HSV with normalization
    # Hue: Use tanh to map mean to [0, 1]
    hue = (np.tanh(w1_mean) + 1.0) / 2.0

    # Saturation: Map std to [0.5, 1.0] for vibrant colors
    saturation = 0.5 + 0.5 * np.tanh(w1_std * 2.0)

    # Value (brightness): Map abs_mean to [0.6, 1.0] for visible colors
    value = 0.6 + 0.4 * np.tanh(w1_abs_mean)

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
        self.species = torch.zeros((size, size), dtype=torch.int32, device=DEVICE)
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
            # Track the hidden size of the best individual's species
            sp_id = self.species[r, c].item()
            self.best_hidden_size = SPECIES_CONFIG[sp_id]['hidden_size']

    # -------------------------------------------------------------------------
    # Genome Computation (Scheme C: Hybrid Genome)
    # -------------------------------------------------------------------------
    def compute_genome(self, y: int, x: int) -> np.ndarray:
        """
        Compute hybrid genome for a cell at position (y, x).

        Returns 12-dimensional genome combining:
        - Neural fingerprint (8-dim): Statistical features from w1, w2
        - Chemical affinity (4-dim): Species chemical preferences

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

        # Get species info
        sp_id = self.species[y, x].item()
        hidden_size = SPECIES_CONFIG[sp_id]['hidden_size']
        chemical_affinity = SPECIES_CONFIG[sp_id]['chemical_affinity']

        # Compute hybrid genome
        return get_full_genome(w1_cell, w2_cell, hidden_size, chemical_affinity)

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

    def update_species_colors_from_genome(self):
        """
        Update species colors based on average genome of population.

        This creates genome-based coloring where similar genomes have similar colors.
        Can be called periodically to update visualization based on evolution.
        """
        # Compute genome for each species (average of all members)
        species_genomes = {}

        for sp_id in range(len(SPECIES_CONFIG)):
            if SPECIES_CONFIG[sp_id].get('extinct', False):
                continue

            # Find all cells of this species
            species_mask = (self.species == sp_id) & self.alive
            if not species_mask.any():
                continue

            # Sample up to 100 cells for efficiency
            positions = species_mask.nonzero(as_tuple=False)
            if len(positions) > 100:
                indices = torch.randperm(len(positions))[:100]
                positions = positions[indices]

            # Compute average genome
            genomes = []
            for pos in positions:
                y, x = pos[0].item(), pos[1].item()
                genome = self.compute_genome(y, x)
                if genome is not None:
                    genomes.append(genome)

            if genomes:
                avg_genome = np.mean(genomes, axis=0)
                species_genomes[sp_id] = avg_genome

                # Update color based on genome
                new_color = genome_to_color(avg_genome)
                SPECIES_CONFIG[sp_id]['color'] = new_color

        return species_genomes

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

        # Local chemical concentrations (transpose for correct indexing)
        local_chemicals = self.chemicals[:, :, :].permute(1, 2, 0)  # [H, W, NUM_CHEMICALS]

        inputs = torch.cat([
            neighbor_energy,                                    # 8: neighbor energy levels
            same_count / 8.0,                                   # 1: same species count
            diff_count / 8.0,                                   # 1: different species count
            energy_norm.unsqueeze(-1),                          # 1: own energy
            (neighbor_alive.sum(dim=-1) / 8.0).unsqueeze(-1),   # 1: total neighbor count
            torch.zeros((self.size, self.size, 8), device=DEVICE),  # 8: padding
            local_chemicals * CHEMICAL_INPUT_WEIGHT,            # NUM_CHEMICALS: local chemical signals
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
        """Cells secrete chemicals based on their species' chemical affinity."""
        if not self.alive.any():
            return

        # For each alive cell, add its chemical signature
        alive_mask = self.alive.cpu().numpy()
        species_mask = self.species.cpu().numpy()

        for chem_id in range(NUM_CHEMICALS):
            secretion_map = torch.zeros((self.size, self.size), dtype=torch.float32, device=DEVICE)

            for sp_id in range(len(SPECIES_CONFIG)):
                if SPECIES_CONFIG[sp_id]['extinct']:
                    continue

                # Get species-specific secretion for this chemical
                affinity = SPECIES_CONFIG[sp_id]['chemical_affinity'][chem_id]
                sp_cells = alive_mask & (species_mask == sp_id)

                if sp_cells.any():
                    # Positive affinity = secretion, negative = absorption
                    if affinity > 0:
                        secretion_map[torch.from_numpy(sp_cells).to(DEVICE)] += CHEMICAL_SECRETION * affinity

            self.chemicals[chem_id] += secretion_map

        # Clamp to prevent overflow
        self.chemicals.clamp_(0, 10.0)

    def step(self):
        """Execute one simulation step (CUDA optimized)."""
        self.is_newborn.fill_(False)

        # Chemical diffusion and secretion
        self._diffuse_chemicals()
        self._secrete_chemicals()

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

        # Update species colors based on genome (if enabled)
        if GENOME_BASED_COLOR and self.generation > 0 and self.generation % GENOME_COLOR_UPDATE_INTERVAL == 0:
            self.update_species_colors_from_genome()

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
                # If delay is 0, speciate immediately without waiting
                if BLOB_SEPARATION_DELAY > 0:
                    continue

            # Check if separated long enough
            if BLOB_SEPARATION_DELAY > 0:
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

            if BLOB_SEPARATION_DELAY > 0:
                separation_duration = self.generation - self.blob_separation_tracker[sp_id]
                print(f"\n[ISOLATION] Geographic separation in {parent_name}")
                print(f"            Blob split after {separation_duration} generations apart")
            else:
                print(f"\n[ISOLATION] Geographic separation detected in {parent_name}")
                print(f"            Immediate speciation triggered")

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
                # Clear blob separation tracker for extinct species
                if sp_id in self.blob_separation_tracker:
                    del self.blob_separation_tracker[sp_id]

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
