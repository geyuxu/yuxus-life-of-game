#!/usr/bin/env python3
"""
Yuxu's Game of Life - High-performance Pygame Renderer

Features:
- 60+ FPS real-time rendering
- Chemical field overlay visualization
- Keyboard controls and camera zoom/pan
- Grid overlay and species statistics
- Much faster than matplotlib for large grids

Controls:
- SPACE: Pause/Resume
- R: Reset simulation
- S: Save best network manually
- C: Toggle chemical overlay
- G: Toggle grid
- +/-: Adjust simulation speed
- Mouse wheel: Zoom
- Click+Drag: Pan camera
- ESC: Quit
"""

import pygame
import numpy as np
import sys
import os
import time
from collections import deque
from scipy.ndimage import label

# Import simulation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evolution import GPULifeGame, genome_to_color
from config import NUM_CHEMICALS, MATE_GENOME_THRESHOLD

# =============================================================================
# CONSTANTS
# =============================================================================
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
FPS_TARGET = 60

# UI Colors (Dark theme)
COLOR_BG = (0, 0, 0)            # Black background
COLOR_TEXT = (220, 220, 220)    # Light text for readability on dark background
COLOR_GRID = (40, 40, 40)       # Dark gray grid
COLOR_PANEL = (20, 20, 20)      # Very dark gray panel

# =============================================================================
# PYGAME RENDERER
# =============================================================================

class PyGameRenderer:
    def __init__(self, game):
        pygame.init()
        pygame.display.set_caption("Yuxu's Game of Life")

        self.game = game
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

        # View settings
        self.show_chemical = False
        self.show_grid = False  # Grid disabled by default (press G to toggle)
        self.chemical_alpha = 0.5
        self.simulation_speed = 1  # Steps per frame

        # Camera
        self.camera_zoom = 1.0
        self.camera_offset = [0, 0]
        self.dragging = False
        self.drag_start = None

        # Stats
        self.fps_history = deque(maxlen=30)
        self.step_time_history = deque(maxlen=30)

        # Genome clustering cache (expensive operation)
        self.cached_clusters = []
        self.cluster_update_interval = 30  # Update clusters every N frames
        self.frames_since_cluster_update = 0

        # Panels
        self.main_panel_width = WINDOW_WIDTH - 300
        self.stats_panel_x = self.main_panel_width
        self.stats_panel_width = 300

        # Calculate cell size
        self.calculate_cell_size()

        # Pre-create surfaces for better performance
        self.main_surface = pygame.Surface((self.main_panel_width, WINDOW_HEIGHT))
        self.stats_surface = pygame.Surface((self.stats_panel_width, WINDOW_HEIGHT))

        # High-resolution render surface for smooth rendering (2x resolution)
        self.render_scale = 2  # Render at 2x resolution for smoothing
        self.hires_surface = pygame.Surface(
            (self.main_panel_width * self.render_scale, WINDOW_HEIGHT * self.render_scale)
        )

        print("\n" + "="*60)
        print("Pygame Renderer Started")
        print("="*60)
        print("\nControls:")
        print("  SPACE  : Pause/Resume")
        print("  R      : Reset simulation")
        print("  S      : Save best network manually")
        print("  C      : Toggle chemical overlay")
        print("  G      : Toggle grid")
        print("  +/-    : Adjust speed")
        print("  Wheel  : Zoom")
        print("  Drag   : Pan camera")
        print("  ESC    : Quit")
        print("="*60 + "\n")

    def calculate_cell_size(self):
        """Calculate optimal cell size based on grid and window size."""
        available_width = self.main_panel_width
        available_height = WINDOW_HEIGHT - 100  # Leave room for header

        cell_size_w = available_width / self.game.size
        cell_size_h = available_height / self.game.size

        self.cell_size = min(cell_size_w, cell_size_h) * self.camera_zoom
        self.grid_offset_x = (available_width - self.cell_size * self.game.size) / 2
        self.grid_offset_y = 50 + (available_height - self.cell_size * self.game.size) / 2

    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        sx = self.grid_offset_x + x * self.cell_size + self.camera_offset[0]
        sy = self.grid_offset_y + y * self.cell_size + self.camera_offset[1]
        return sx, sy

    def screen_to_world(self, sx, sy):
        """Convert screen coordinates to world coordinates."""
        wx = (sx - self.grid_offset_x - self.camera_offset[0]) / self.cell_size
        wy = (sy - self.grid_offset_y - self.camera_offset[1]) / self.cell_size
        return int(wx), int(wy)

    def render_environment(self):
        """Render the main simulation environment with smooth anti-aliasing."""
        # Clear high-resolution surface
        self.hires_surface.fill(COLOR_BG)

        # Get data from game
        alive = self.game.alive.cpu().numpy()
        energy = self.game.energy.cpu().numpy()
        genome = self.game.genome.cpu().numpy()  # Still needed for neighbor similarity checks
        colors = self.game.cell_colors  # Use cached colors instead of recomputing

        # Scaling factor for high-res rendering
        scale = self.render_scale

        # Render cells as filled rectangles for connected tissue appearance
        for y in range(self.game.size):
            for x in range(self.game.size):
                if not alive[y, x]:
                    continue

                # Get cached genome-based color (updated periodically for efficiency)
                base_color = colors[y, x]
                cell_genome = genome[y, x]  # Needed for neighbor similarity checks

                # Modulate by energy (brightness)
                energy_norm = min(1.0, energy[y, x] / 100.0)
                color = tuple(int(c * 255 * energy_norm) for c in base_color)

                # Calculate screen position (scaled for high-res)
                sx, sy = self.world_to_screen(x, y)
                sx_hires = int(sx * scale)
                sy_hires = int(sy * scale)
                cell_size_hires = int(self.cell_size * scale)

                # Draw cell as filled rectangle on high-res surface
                rect = pygame.Rect(sx_hires, sy_hires, cell_size_hires, cell_size_hires)
                pygame.draw.rect(self.hires_surface, color, rect)

                # Draw subtle internal borders for same-species neighbors (visual texture)
                border_color = tuple(max(0, c - 15) for c in color)

                # Check if neighbors are similar species
                dirs = [(1, 0), (0, 1)]  # Right and bottom
                for dx, dy in dirs:
                    nx = (x + dx) % self.game.size
                    ny = (y + dy) % self.game.size

                    if alive[ny, nx]:
                        neighbor_genome = genome[ny, nx]
                        genome_dist = np.linalg.norm(cell_genome - neighbor_genome)

                        # Same species: draw subtle internal line
                        if genome_dist < MATE_GENOME_THRESHOLD:
                            if dx == 1:  # Right neighbor
                                line_x = sx_hires + cell_size_hires
                                pygame.draw.line(self.hires_surface, border_color,
                                               (line_x, sy_hires), (line_x, sy_hires + cell_size_hires), scale)
                            else:  # Bottom neighbor
                                line_y = sy_hires + cell_size_hires
                                pygame.draw.line(self.hires_surface, border_color,
                                               (sx_hires, line_y), (sx_hires + cell_size_hires, line_y), scale)

        # Organization boundaries removed for cleaner appearance
        # (Previously drew red boundaries between different species)

        # Smooth scale down to main surface (this provides anti-aliasing)
        pygame.transform.smoothscale(self.hires_surface, self.main_surface.get_size(), self.main_surface)

        # Chemical overlay
        if self.show_chemical:
            self.render_chemical_overlay()

        # Grid overlay
        if self.show_grid and self.cell_size > 3:
            self.render_grid()

        # Copy to main window
        self.window.blit(self.main_surface, (0, 0))

    def render_chemical_overlay(self):
        """Render chemical field as colored overlay."""
        chemicals = self.game.chemicals.cpu().numpy()

        # Create overlay surface with alpha
        overlay = pygame.Surface((self.main_panel_width, WINDOW_HEIGHT))
        overlay.set_alpha(int(255 * self.chemical_alpha))
        overlay.fill(COLOR_BG)

        # Render each chemical with different color
        chemical_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0)   # Yellow
        ]

        for y in range(self.game.size):
            for x in range(self.game.size):
                # Combine all chemicals
                total_intensity = 0
                combined_color = [0, 0, 0]

                for chem_id in range(NUM_CHEMICALS):
                    concentration = chemicals[chem_id, y, x]
                    if concentration > 0.01:
                        intensity = min(1.0, concentration / 2.0)  # Normalize
                        total_intensity += intensity

                        for i in range(3):
                            combined_color[i] += chemical_colors[chem_id][i] * intensity

                if total_intensity > 0:
                    # Normalize combined color
                    combined_color = [min(255, int(c / total_intensity)) for c in combined_color]

                    # Draw colored cell as circle (consistent with cell rendering)
                    sx, sy = self.world_to_screen(x, y)
                    center_x = int(sx + self.cell_size / 2)
                    center_y = int(sy + self.cell_size / 2)
                    radius = max(1, int(self.cell_size * 0.45))
                    pygame.draw.circle(overlay, combined_color, (center_x, center_y), radius)

        self.main_surface.blit(overlay, (0, 0))

    def render_grid(self):
        """Render grid lines with adaptive opacity based on cell size."""
        # Adjust grid opacity: darker when cells are small, lighter when cells are large
        # For cell_size < 10: use darker grid (60, 60, 60)
        # For cell_size > 30: use lighter grid (40, 40, 40) or less
        if self.cell_size > 30:
            # Large cells: very subtle grid
            grid_intensity = max(25, int(60 - (self.cell_size - 30) * 0.8))
        elif self.cell_size > 15:
            # Medium cells: moderate grid
            grid_intensity = int(60 - (self.cell_size - 15) * 0.5)
        else:
            # Small cells: standard grid
            grid_intensity = 60

        grid_color = (grid_intensity, grid_intensity, grid_intensity)

        for i in range(self.game.size + 1):
            # Vertical lines
            sx, sy_top = self.world_to_screen(i, 0)
            _, sy_bottom = self.world_to_screen(i, self.game.size)
            pygame.draw.line(self.main_surface, grid_color, (sx, sy_top), (sx, sy_bottom), 1)

            # Horizontal lines
            sx_left, sy = self.world_to_screen(0, i)
            sx_right, _ = self.world_to_screen(self.game.size, i)
            pygame.draw.line(self.main_surface, grid_color, (sx_left, sy), (sx_right, sy), 1)

    def cluster_genomes(self, max_cells=500):
        """Cluster alive cells by genome similarity to find emergent species.

        Uses sampling to find cluster centers, then assigns all cells.
        """
        alive = self.game.alive.cpu().numpy()
        genome = self.game.genome.cpu().numpy()

        if not alive.any():
            return []

        # Get all alive cell genomes
        alive_genomes = genome[alive]  # [N_total, 12]
        N_total = len(alive_genomes)

        if N_total == 0:
            return []

        # Step 1: Sample for clustering (if needed)
        if N_total > max_cells:
            sample_indices = np.random.choice(N_total, max_cells, replace=False)
            sample_genomes = alive_genomes[sample_indices]
            N_sample = max_cells
        else:
            sample_genomes = alive_genomes
            N_sample = N_total

        # Step 2: Build adjacency matrix on sample
        adjacency = np.zeros((N_sample, N_sample), dtype=bool)
        for i in range(N_sample):
            for j in range(i+1, N_sample):
                dist = np.linalg.norm(sample_genomes[i] - sample_genomes[j])
                if dist < MATE_GENOME_THRESHOLD:
                    adjacency[i, j] = True
                    adjacency[j, i] = True

        # Step 3: Find cluster centers from sample using BFS
        cluster_centers = []
        visited = np.zeros(N_sample, dtype=bool)

        for i in range(N_sample):
            if visited[i]:
                continue

            # BFS to find all cells in this cluster
            cluster_indices = []
            queue = [i]
            visited[i] = True

            while queue:
                current = queue.pop(0)
                cluster_indices.append(current)

                # Add unvisited neighbors
                neighbors = np.where(adjacency[current] & ~visited)[0]
                for neighbor in neighbors:
                    visited[neighbor] = True
                    queue.append(neighbor)

            if cluster_indices:
                # Compute cluster center (mean genome)
                cluster_genomes = sample_genomes[cluster_indices]
                center = cluster_genomes.mean(axis=0)
                cluster_centers.append(center)

        if len(cluster_centers) == 0:
            return []

        # Step 4: Assign ALL cells to nearest cluster
        cluster_centers_array = np.array(cluster_centers)  # [K, 12]
        cluster_counts = [0] * len(cluster_centers)

        for genome_vec in alive_genomes:
            # Find nearest cluster center
            distances = np.linalg.norm(cluster_centers_array - genome_vec, axis=1)
            nearest_cluster = np.argmin(distances)
            cluster_counts[nearest_cluster] += 1

        # Step 5: Build cluster info
        clusters = []
        for idx, center in enumerate(cluster_centers):
            if cluster_counts[idx] > 0:
                clusters.append({
                    'size': cluster_counts[idx],
                    'color': genome_to_color(center),
                    'avg_genome': center
                })

        # Sort by size (largest first)
        clusters.sort(key=lambda x: x['size'], reverse=True)
        return clusters

    def render_stats_panel(self):
        """Render statistics panel on the right."""
        self.stats_surface.fill(COLOR_PANEL)

        y_offset = 20

        # Header
        title = self.font_large.render('Statistics', True, COLOR_TEXT)
        self.stats_surface.blit(title, (10, y_offset))
        y_offset += 50

        # Basic stats
        total_pop = self.game.history['population'][-1] if self.game.history['population'] else 0

        # Update genome clusters periodically (expensive operation)
        self.frames_since_cluster_update += 1
        if self.frames_since_cluster_update >= self.cluster_update_interval:
            self.cached_clusters = self.cluster_genomes()
            self.frames_since_cluster_update = 0

        clusters = self.cached_clusters
        num_clusters = len(clusters)

        # Get validation stats
        val_stats = self.game.get_validation_stats()

        stats_lines = [
            f"Generation: {self.game.generation:,}",
            f"Population: {total_pop:,}",
            f"Emergent Groups: {num_clusters}",
            f"",
            f"FPS: {int(np.mean(self.fps_history)) if self.fps_history else 0}",
            f"Step: {int(np.mean(self.step_time_history)) if self.step_time_history else 0}ms",
            f"Speed: {self.simulation_speed}x",
            f"",
            f"Zoom: {self.camera_zoom:.2f}x",
            f"Chemical: {'ON' if self.show_chemical else 'OFF'}",
            f"Grid: {'ON' if self.show_grid else 'OFF'}",
        ]

        # Add validation stats if available
        if val_stats:
            total = val_stats['gen0_count'] + val_stats['gen1_5_count'] + val_stats['gen6plus_count'] + val_stats['random_count']
            if total > 0:
                stats_lines.extend([
                    f"",
                    f"--- Lineage Tracking ---",
                    f"Trained: {val_stats['trained_total']} ({val_stats['trained_total']/total*100:.0f}%)",
                    f"  Gen0: {val_stats['gen0_count']}",
                    f"  Gen1-5: {val_stats['gen1_5_count']}",
                    f"  Gen6+: {val_stats['gen6plus_count']}",
                    f"Random: {val_stats['random_count']} ({val_stats['random_count']/total*100:.0f}%)",
                    f"",
                    f"Trained vs Random:",
                    f"Lifetime: {val_stats['trained_lineage_avg_lifetime']:.0f} vs {val_stats['random_avg_lifetime']:.0f}",
                ])
                if val_stats['random_avg_lifetime'] > 0 and val_stats['trained_lineage_avg_lifetime'] > 0:
                    ratio = val_stats['trained_lineage_avg_lifetime'] / val_stats['random_avg_lifetime']
                    symbol = "✓" if ratio > 1 else "✗"
                    stats_lines.append(f"{symbol} {ratio:.2f}x performance")

        for line in stats_lines:
            text = self.font_small.render(line, True, COLOR_TEXT)
            self.stats_surface.blit(text, (10, y_offset))
            y_offset += 22

        # Show emergent species groups
        y_offset += 10
        title = self.font_medium.render('Genome Clusters', True, COLOR_TEXT)
        self.stats_surface.blit(title, (10, y_offset))
        y_offset += 30

        for i, cluster in enumerate(clusters[:15]):  # Show top 15
            if y_offset > WINDOW_HEIGHT - 30:
                break

            size = cluster['size']
            pct = (size / total_pop * 100) if total_pop > 0 else 0
            color_tuple = cluster['color']
            color = tuple(int(c * 255) for c in color_tuple)

            # Color square
            rect = pygame.Rect(10, y_offset, 15, 15)
            pygame.draw.rect(self.stats_surface, color, rect)
            pygame.draw.rect(self.stats_surface, COLOR_TEXT, rect, 1)

            # Cluster info
            text = self.font_small.render(f"G{i+1:2d}  {size:5d}  {pct:4.1f}%",
                                         True, COLOR_TEXT)
            self.stats_surface.blit(text, (30, y_offset))
            y_offset += 20

        # Copy to main window
        self.window.blit(self.stats_surface, (self.stats_panel_x, 0))

    def render_header(self):
        """Render header bar."""
        header_rect = pygame.Rect(0, 0, self.main_panel_width, 40)
        pygame.draw.rect(self.window, COLOR_PANEL, header_rect)

        title = self.font_large.render('🧬 Neuroevolution Arena', True, COLOR_TEXT)
        self.window.blit(title, (10, 5))

        # Status
        status = "PAUSED" if self.simulation_speed == 0 else "RUNNING"
        status_color = (255, 200, 0) if self.simulation_speed == 0 else (0, 255, 100)
        status_text = self.font_medium.render(status, True, status_color)
        self.window.blit(status_text, (self.main_panel_width - 150, 8))

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.simulation_speed = 0 if self.simulation_speed > 0 else 1
                elif event.key == pygame.K_r:
                    self.game = GPULifeGame()
                    print("Simulation reset")
                elif event.key == pygame.K_c:
                    self.show_chemical = not self.show_chemical
                    print(f"Chemical overlay: {'ON' if self.show_chemical else 'OFF'}")
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                    print(f"Grid: {'ON' if self.show_grid else 'OFF'}")
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.simulation_speed = min(10, self.simulation_speed + 1)
                    print(f"Speed: {self.simulation_speed}x")
                elif event.key == pygame.K_MINUS:
                    self.simulation_speed = max(0, self.simulation_speed - 1)
                    print(f"Speed: {self.simulation_speed}x")
                elif event.key == pygame.K_s:
                    # Manual save
                    self.game._update_best_network()
                    self.game._save_best_weights()
                    print(f"[MANUAL SAVE] Gen {self.game.generation}, Fitness={self.game.best_fitness:.1f}")

            elif event.type == pygame.MOUSEWHEEL:
                # Zoom
                old_zoom = self.camera_zoom
                self.camera_zoom *= 1.1 if event.y > 0 else 0.9
                self.camera_zoom = max(0.1, min(5.0, self.camera_zoom))
                self.calculate_cell_size()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.dragging = True
                    self.drag_start = event.pos

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    dx = event.pos[0] - self.drag_start[0]
                    dy = event.pos[1] - self.drag_start[1]
                    self.camera_offset[0] += dx
                    self.camera_offset[1] += dy
                    self.drag_start = event.pos

        return True

    def run(self):
        """Main rendering loop."""
        running = True
        paused = False

        while running:
            frame_start = time.time()

            # Handle events
            running = self.handle_events()

            # Run simulation steps
            if self.simulation_speed > 0:
                step_start = time.time()
                for _ in range(self.simulation_speed):
                    self.game.step()
                    if self.game.is_extinct():
                        print("All organisms extinct!")
                        self.simulation_speed = 0
                        break
                step_time = (time.time() - step_start) * 1000
                self.step_time_history.append(step_time)

            # Render
            self.render_header()
            self.render_environment()
            self.render_stats_panel()

            pygame.display.flip()

            # Track FPS
            frame_time = time.time() - frame_start
            self.fps_history.append(1.0 / frame_time if frame_time > 0 else 0)

            self.clock.tick(FPS_TARGET)

        pygame.quit()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("Initializing simulation...")
    game = GPULifeGame()

    print("Starting renderer...")
    renderer = PyGameRenderer(game)
    renderer.run()
