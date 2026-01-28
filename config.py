"""
Configuration file for Neuroevolution Arena.

All simulation parameters can be adjusted here.
"""

import os

# ==============================================================================
# SIMULATION SETTINGS
# ==============================================================================

# Grid and initial conditions
GRID_SIZE = 100                 # Size of the simulation grid (100x100)
INITIAL_ENERGY = 30.0          # Starting energy for new cells
MAX_ENERGY = 100.0             # Maximum energy a cell can have

# ==============================================================================
# CELL BEHAVIOR
# ==============================================================================

# Movement and crowding
MOVE_COST = 0.2                # Energy cost per move
CROWDING_THRESHOLD = 4         # Neighbors before crowding penalty applies
CROWDING_PENALTY = 0.5         # Energy penalty per excess neighbor

# Metabolism and survival
SPECIES_METABOLISM = 0.1       # Energy consumed per step
SPECIES_STARVATION = 100       # Steps without food before death

# Reproduction
SPECIES_REPRO_THRESHOLD = 20   # Minimum energy to reproduce
SPECIES_REPRO_COST = 8         # Energy cost of reproduction
SPECIES_OFFSPRING_ENERGY = 25  # Starting energy for offspring

# Combat
ATTACK_BONUS = 1.2             # Energy multiplier when eating prey

# ==============================================================================
# NEURAL NETWORK ARCHITECTURE
# ==============================================================================

NUM_CHEMICALS = 4              # Number of chemical signal types
INPUT_SIZE = 20 + NUM_CHEMICALS  # NN input size (20 base + chemicals)
NUM_ACTIONS = 7                # Number of possible actions

# Actions
ACTION_STAY = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_EAT = 5
ACTION_REPRODUCE = 6

# Hidden layer configuration
SPECIES_HIDDEN_SIZE = 8        # Default hidden layer size
MAX_HIDDEN_SIZE = 240          # Maximum hidden neurons
MIN_HIDDEN_SIZE = 4            # Minimum hidden neurons
HIDDEN_SIZE_INCREMENT = 2      # Neurons added on mutation

# ==============================================================================
# CHEMICAL SIGNALING
# ==============================================================================

CHEMICAL_DIFFUSION = 0.3       # Diffusion rate (0-1)
CHEMICAL_DECAY = 0.05          # Decay rate per step
CHEMICAL_SECRETION = 0.1       # Amount secreted per step
CHEMICAL_INPUT_WEIGHT = 0.2    # Chemical influence on NN input

# ==============================================================================
# EVOLUTION AND MUTATION
# ==============================================================================

# Basic mutation
MUTATION_RATE = 0.1            # Base mutation rate for weights

# Genome-based mating
MATE_GENOME_THRESHOLD = 0.5    # Max genome distance for mating (lower = stricter)

# Dominance-based forced mutation
DOMINANCE_THRESHOLD = 0.3      # Force mutation when species > 30% of population
DOMINANCE_CHECK_INTERVAL = 100 # Check dominance every N generations

# Legacy species system (mostly unused in genome-based system)
INITIAL_NUM_SPECIES = 3
MAX_SPECIES = 500
MAX_ACTIVE_SPECIES = 50
MUTATION_INTERVAL = 100
RANDOM_MUTATION_CHANCE = 0.05
EXTINCT_RECYCLE_DELAY = 50
BLOB_SEPARATION_DELAY = 0
BLOB_CHECK_INTERVAL = 200

# ==============================================================================
# REINFORCEMENT LEARNING
# ==============================================================================

RL_LEARNING_RATE = 0.01        # Learning rate for policy gradient
REWARD_EAT_PREY = 2.0          # Reward for successful hunt
REWARD_SURVIVE_ATTACK = 1.0    # Reward for escaping
REWARD_REPRODUCE = 1.5         # Reward for reproduction

# Experience Replay
RL_REPLAY_BUFFER_SIZE = 10000  # Maximum experiences to store
RL_REPLAY_BATCH_SIZE = 256     # Batch size for replay sampling
RL_USE_REPLAY = True           # Enable experience replay (improves learning stability)

# ==============================================================================
# FITNESS FUNCTION
# ==============================================================================

# Multi-objective fitness weights
FITNESS_WEIGHT_LIFETIME = 1.0       # Survival time
FITNESS_WEIGHT_REPRODUCTION = 10.0  # Reproductive success (harder to achieve)
FITNESS_WEIGHT_DIVERSITY = 5.0      # Genetic uniqueness bonus
FITNESS_WEIGHT_ENERGY = 0.5         # Energy efficiency

# ==============================================================================
# NETWORK PERSISTENCE
# ==============================================================================

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FILE = os.path.join(SAVE_DIR, "best_brain.pt")

# Auto-save settings
AUTO_SAVE_ENABLED = True       # Enable automatic saving
SAVE_INTERVAL_SECONDS = 120    # Auto-save every 2 minutes
SAVE_INTERVAL = 50             # Legacy: generation-based interval

# Elite inheritance for validation
ELITE_RATIO = 0.01             # 10% trained, 90% random (strict validation)

# ==============================================================================
# VISUALIZATION
# ==============================================================================

# Genome-based coloring
GENOME_BASED_COLOR = True      # Color cells by genome similarity
GENOME_COLOR_UPDATE_INTERVAL = 10  # Update colors every N generations (smaller = smoother color transitions)

# Performance
HISTORY_WINDOW = 1000          # Keep last N generations (0 = unlimited)
HISTORY_UPDATE_INTERVAL = 10   # Update history stats every N generations (higher = faster)
RENDER_INTERVAL = 1            # Render every N frames
CHART_UPDATE_INTERVAL = 5      # Update charts every N frames

# Debug
VERBOSE_SPECIATION = False     # Print detailed speciation logs

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def print_config():
    """Print current configuration."""
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)

    print(f"\n[Grid & Energy]")
    print(f"  Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Initial energy: {INITIAL_ENERGY}, Max: {MAX_ENERGY}")
    print(f"  Metabolism: {SPECIES_METABOLISM}/step")

    print(f"\n[Reproduction]")
    print(f"  Threshold: {SPECIES_REPRO_THRESHOLD} energy")
    print(f"  Cost: {SPECIES_REPRO_COST}")
    print(f"  Offspring energy: {SPECIES_OFFSPRING_ENERGY}")

    print(f"\n[Neural Network]")
    print(f"  Input size: {INPUT_SIZE}")
    print(f"  Hidden neurons: {MIN_HIDDEN_SIZE}-{MAX_HIDDEN_SIZE} (default {SPECIES_HIDDEN_SIZE})")
    print(f"  Output actions: {NUM_ACTIONS}")

    print(f"\n[Evolution]")
    print(f"  Mutation rate: {MUTATION_RATE}")
    print(f"  Mate genome threshold: {MATE_GENOME_THRESHOLD}")
    print(f"  Dominance threshold: {DOMINANCE_THRESHOLD*100:.0f}%")

    print(f"\n[Fitness Function]")
    print(f"  Lifetime weight: {FITNESS_WEIGHT_LIFETIME}")
    print(f"  Reproduction weight: {FITNESS_WEIGHT_REPRODUCTION}")
    print(f"  Diversity weight: {FITNESS_WEIGHT_DIVERSITY}")
    print(f"  Energy weight: {FITNESS_WEIGHT_ENERGY}")

    print(f"\n[Auto-save]")
    print(f"  Enabled: {AUTO_SAVE_ENABLED}")
    print(f"  Interval: {SAVE_INTERVAL_SECONDS}s")
    print(f"  Elite ratio: {ELITE_RATIO*100:.0f}% trained, {(1-ELITE_RATIO)*100:.0f}% random")

    print("="*70 + "\n")

if __name__ == "__main__":
    print_config()
