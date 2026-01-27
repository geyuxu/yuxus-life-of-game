#!/usr/bin/env python3
"""
High-performance Pygame renderer for the neuroevolution simulation.

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

# Import simulation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import GPULifeGame, SPECIES_CONFIG, NUM_CHEMICALS

# =============================================================================
# CONSTANTS
# =============================================================================
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
FPS_TARGET = 60

# UI Colors
COLOR_BG = (20, 20, 20)
COLOR_TEXT = (220, 220, 220)
COLOR_GRID = (60, 60, 60)
COLOR_PANEL = (30, 30, 30)

# =============================================================================
# PYGAME RENDERER
# =============================================================================

class PyGameRenderer:
    def __init__(self, game):
        pygame.init()
        pygame.display.set_caption('Neuroevolution Arena - Pygame Renderer')

        self.game = game
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

        # View settings
        self.show_chemical = False
        self.show_grid = True
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

        # Panels
        self.main_panel_width = WINDOW_WIDTH - 300
        self.stats_panel_x = self.main_panel_width
        self.stats_panel_width = 300

        # Calculate cell size
        self.calculate_cell_size()

        # Pre-create surfaces for better performance
        self.main_surface = pygame.Surface((self.main_panel_width, WINDOW_HEIGHT))
        self.stats_surface = pygame.Surface((self.stats_panel_width, WINDOW_HEIGHT))

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
        """Render the main simulation environment."""
        self.main_surface.fill(COLOR_BG)

        # Get data from game
        alive = self.game.alive.cpu().numpy()
        energy = self.game.energy.cpu().numpy()
        species = self.game.species.cpu().numpy()

        # Render cells
        for y in range(self.game.size):
            for x in range(self.game.size):
                if not alive[y, x]:
                    continue

                # Get species color
                sp_id = species[y, x]
                if sp_id >= len(SPECIES_CONFIG):
                    continue

                base_color = SPECIES_CONFIG[sp_id]['color']

                # Modulate by energy (brightness)
                energy_norm = min(1.0, energy[y, x] / 100.0)
                color = tuple(int(c * 255 * energy_norm) for c in base_color)

                # Calculate screen position
                sx, sy = self.world_to_screen(x, y)

                # Draw cell
                rect = pygame.Rect(sx, sy, self.cell_size, self.cell_size)
                pygame.draw.rect(self.main_surface, color, rect)

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

                    # Draw colored cell
                    sx, sy = self.world_to_screen(x, y)
                    rect = pygame.Rect(sx, sy, self.cell_size, self.cell_size)
                    pygame.draw.rect(overlay, combined_color, rect)

        self.main_surface.blit(overlay, (0, 0))

    def render_grid(self):
        """Render grid lines."""
        for i in range(self.game.size + 1):
            # Vertical lines
            sx, sy_top = self.world_to_screen(i, 0)
            _, sy_bottom = self.world_to_screen(i, self.game.size)
            pygame.draw.line(self.main_surface, COLOR_GRID, (sx, sy_top), (sx, sy_bottom), 1)

            # Horizontal lines
            sx_left, sy = self.world_to_screen(0, i)
            sx_right, _ = self.world_to_screen(self.game.size, i)
            pygame.draw.line(self.main_surface, COLOR_GRID, (sx_left, sy), (sx_right, sy), 1)

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
        alive_species = sum(1 for sp in SPECIES_CONFIG if not sp.get('extinct', False))

        stats_lines = [
            f"Generation: {self.game.generation:,}",
            f"Species: {alive_species}",
            f"Population: {total_pop:,}",
            f"",
            f"FPS: {int(np.mean(self.fps_history)) if self.fps_history else 0}",
            f"Step: {int(np.mean(self.step_time_history)) if self.step_time_history else 0}ms",
            f"Speed: {self.simulation_speed}x",
            f"",
            f"Zoom: {self.camera_zoom:.2f}x",
            f"Chemical: {'ON' if self.show_chemical else 'OFF'}",
            f"Grid: {'ON' if self.show_grid else 'OFF'}",
        ]

        for line in stats_lines:
            text = self.font_small.render(line, True, COLOR_TEXT)
            self.stats_surface.blit(text, (10, y_offset))
            y_offset += 22

        # Species list
        y_offset += 20
        title = self.font_medium.render('Species', True, COLOR_TEXT)
        self.stats_surface.blit(title, (10, y_offset))
        y_offset += 30

        # Gather species data
        species_data = []
        for sp_id in range(len(SPECIES_CONFIG)):
            if SPECIES_CONFIG[sp_id].get('extinct', False):
                continue

            if sp_id < len(self.game.history['species']) and self.game.history['species'][sp_id]:
                count = self.game.history['species'][sp_id][-1]
            else:
                count = 0

            if count == 0:
                continue

            name = SPECIES_CONFIG[sp_id]['name']
            pct = (count / total_pop * 100) if total_pop > 0 else 0
            color_tuple = SPECIES_CONFIG[sp_id]['color']
            color = tuple(int(c * 255) for c in color_tuple)

            species_data.append((name, count, pct, color))

        # Sort by population
        species_data.sort(key=lambda x: x[1], reverse=True)

        # Render species entries
        for name, count, pct, color in species_data[:20]:  # Limit to top 20
            if y_offset > WINDOW_HEIGHT - 30:
                break

            # Color square
            rect = pygame.Rect(10, y_offset, 15, 15)
            pygame.draw.rect(self.stats_surface, color, rect)
            pygame.draw.rect(self.stats_surface, COLOR_TEXT, rect, 1)

            # Species info
            text = self.font_small.render(f"{name[:8]:8s} {count:5d} {pct:4.1f}%",
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
