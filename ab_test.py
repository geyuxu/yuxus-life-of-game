#!/usr/bin/env python3
"""
A/B Testing Framework for Neuroevolution Arena

Allows comparing two different configurations by running them in parallel
and tracking comparative metrics.

Usage:
    python3 ab_test.py --generations 1000 --output results.json

Example configurations:
    A: High mutation (0.2), Low RL rate (0.005)
    B: Low mutation (0.05), High RL rate (0.02)
"""

import argparse
import json
import time
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evolution import GPULifeGame
import config


class ABTestRunner:
    """
    Run two simulations in parallel with different configurations and compare results.
    """

    def __init__(self, config_a: Dict, config_b: Dict, test_name: str = "A/B Test"):
        """
        Initialize A/B test runner.

        Args:
            config_a: Configuration dictionary for variant A
            config_b: Configuration dictionary for variant B
            test_name: Name of the test for reporting
        """
        self.config_a = config_a
        self.config_b = config_b
        self.test_name = test_name

        # Initialize simulations
        print(f"\n{'='*70}")
        print(f"INITIALIZING A/B TEST: {test_name}")
        print(f"{'='*70}\n")

        self.game_a = None
        self.game_b = None
        self.metrics_a = []
        self.metrics_b = []

    def _apply_config(self, config_dict: Dict):
        """Apply configuration parameters to config module."""
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown config parameter: {key}")

    def _collect_metrics(self, game: GPULifeGame) -> Dict:
        """
        Collect metrics from a simulation at current generation.

        Returns:
            Dictionary of metrics
        """
        alive_count = game.alive.sum().item()

        if alive_count == 0:
            return {
                'generation': game.generation,
                'population': 0,
                'avg_fitness': 0.0,
                'diversity': 0.0,
                'trained_ratio': 0.0,
                'avg_energy': 0.0,
                'avg_lifetime': 0.0,
                'avg_repro': 0.0,
            }

        # Calculate fitness
        fitness = game._calculate_fitness()
        avg_fitness = fitness[game.alive].mean().item()

        # Calculate diversity (average pairwise distance)
        alive_genomes = game.genome[game.alive]
        if len(alive_genomes) > 1:
            n_sample = min(len(alive_genomes), 100)
            import torch
            sample_indices = torch.randperm(len(alive_genomes), device=game.genome.device)[:n_sample]
            sample_genomes = alive_genomes[sample_indices]

            expanded1 = sample_genomes.unsqueeze(1)
            expanded2 = sample_genomes.unsqueeze(0)
            distances = torch.norm(expanded1 - expanded2, dim=2)

            mask = ~torch.eye(n_sample, dtype=torch.bool, device=game.genome.device)
            avg_diversity = distances[mask].mean().item()
        else:
            avg_diversity = 0.0

        # Trained lineage ratio
        trained_count = (game.alive & (game.trained_generation >= 0)).sum().item()
        trained_ratio = trained_count / alive_count if alive_count > 0 else 0.0

        # Other metrics
        avg_energy = game.energy[game.alive].mean().item()
        avg_lifetime = game.lifetime[game.alive].float().mean().item()
        avg_repro = game.repro_count[game.alive].float().mean().item()

        return {
            'generation': game.generation,
            'population': alive_count,
            'avg_fitness': avg_fitness,
            'diversity': avg_diversity,
            'trained_ratio': trained_ratio,
            'avg_energy': avg_energy,
            'avg_lifetime': avg_lifetime,
            'avg_repro': avg_repro,
        }

    def run(self, generations: int = 1000, report_interval: int = 100):
        """
        Run the A/B test for specified number of generations.

        Args:
            generations: Number of generations to run
            report_interval: Print progress every N generations
        """
        print("Variant A Configuration:")
        for key, value in self.config_a.items():
            print(f"  {key}: {value}")

        print("\nVariant B Configuration:")
        for key, value in self.config_b.items():
            print(f"  {key}: {value}")

        print(f"\nRunning {generations} generations...\n")

        # Initialize variant A
        print("Initializing Variant A...")
        self._apply_config(self.config_a)
        self.game_a = GPULifeGame()

        # Initialize variant B
        print("Initializing Variant B...")
        self._apply_config(self.config_b)
        self.game_b = GPULifeGame()

        # Run both simulations
        start_time = time.time()

        for gen in range(generations):
            # Step variant A
            self._apply_config(self.config_a)
            self.game_a.step()

            # Step variant B
            self._apply_config(self.config_b)
            self.game_b.step()

            # Collect metrics at intervals
            if gen % report_interval == 0:
                metrics_a = self._collect_metrics(self.game_a)
                metrics_b = self._collect_metrics(self.game_b)

                self.metrics_a.append(metrics_a)
                self.metrics_b.append(metrics_b)

                elapsed = time.time() - start_time
                gens_per_sec = (gen + 1) / elapsed if elapsed > 0 else 0

                print(f"Gen {gen:4d} | "
                      f"A: Pop={metrics_a['population']:4d} Fit={metrics_a['avg_fitness']:6.1f} Div={metrics_a['diversity']:.2f} | "
                      f"B: Pop={metrics_b['population']:4d} Fit={metrics_b['avg_fitness']:6.1f} Div={metrics_b['diversity']:.2f} | "
                      f"{gens_per_sec:.1f} gen/s")

        # Final metrics
        final_a = self._collect_metrics(self.game_a)
        final_b = self._collect_metrics(self.game_b)

        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"TEST COMPLETE - {elapsed:.1f}s total")
        print(f"{'='*70}\n")

        return final_a, final_b

    def print_results(self, final_a: Dict, final_b: Dict):
        """Print comparison results."""
        print(f"\nFINAL RESULTS ({self.test_name})")
        print(f"{'='*70}\n")

        metrics = ['population', 'avg_fitness', 'diversity', 'trained_ratio', 'avg_energy', 'avg_lifetime', 'avg_repro']

        print(f"{'Metric':<20} {'Variant A':>15} {'Variant B':>15} {'Winner':>10}")
        print(f"{'-'*70}")

        for metric in metrics:
            val_a = final_a[metric]
            val_b = final_b[metric]

            winner = "A" if val_a > val_b else "B" if val_b > val_a else "Tie"
            winner_symbol = "✓" if winner != "Tie" else "="

            print(f"{metric:<20} {val_a:>15.2f} {val_b:>15.2f} {winner:>8s} {winner_symbol}")

        print(f"\n{'='*70}\n")

    def save_results(self, filepath: str, final_a: Dict, final_b: Dict):
        """Save test results to JSON file."""
        results = {
            'test_name': self.test_name,
            'config_a': self.config_a,
            'config_b': self.config_b,
            'final_metrics_a': final_a,
            'final_metrics_b': final_b,
            'metrics_history_a': self.metrics_a,
            'metrics_history_b': self.metrics_b,
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {filepath}")


def main():
    """Main entry point for A/B testing."""
    parser = argparse.ArgumentParser(description='A/B Testing Framework for Neuroevolution Arena')
    parser.add_argument('--generations', '-g', type=int, default=1000,
                        help='Number of generations to run (default: 1000)')
    parser.add_argument('--report-interval', '-r', type=int, default=100,
                        help='Report progress every N generations (default: 100)')
    parser.add_argument('--output', '-o', type=str, default='ab_test_results.json',
                        help='Output file for results (default: ab_test_results.json)')
    parser.add_argument('--test-name', '-n', type=str, default='Mutation vs RL Rate',
                        help='Name of the test (default: "Mutation vs RL Rate")')

    args = parser.parse_args()

    # Example configuration: Test mutation rate vs RL learning rate
    config_a = {
        'MUTATION_RATE': 0.2,           # High mutation
        'RL_LEARNING_RATE': 0.005,      # Low RL rate
        'SPECIES_METABOLISM': 0.1,
    }

    config_b = {
        'MUTATION_RATE': 0.05,          # Low mutation
        'RL_LEARNING_RATE': 0.02,       # High RL rate
        'SPECIES_METABOLISM': 0.1,
    }

    # Run test
    runner = ABTestRunner(config_a, config_b, args.test_name)
    final_a, final_b = runner.run(generations=args.generations, report_interval=args.report_interval)

    # Print and save results
    runner.print_results(final_a, final_b)
    runner.save_results(args.output, final_a, final_b)


if __name__ == '__main__':
    main()
