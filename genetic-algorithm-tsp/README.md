# 3D Traveling Salesman Problem — Genetic Algorithm

This project implements an optimization agent that solves a 3D Traveling Salesman Problem (TSP) using a Genetic Algorithm built entirely from scratch.

## Overview
The agent searches for near-optimal routes across a large combinatorial space of 3D locations, balancing exploration and exploitation through population-based learning.

## Key Features
- Full genetic algorithm pipeline:
  - Population initialization
  - Fitness evaluation (inverse of total distance)
  - Roulette-wheel selection with elitism
  - Two-point crossover
  - Multi-strategy mutation
- Hybrid initialization using random routes and nearest-neighbor heuristics
- Designed to scale across large search spaces efficiently

## Technical Details
- Language: Python
- Algorithms: Genetic Algorithm, heuristic optimization
- Scale:
  - 100+ candidate solutions per generation
  - ~500 generations per run
- Distance metric: 3D Euclidean distance

## What This Demonstrates
- Planning and optimization in large, non-convex search spaces
- Iterative improvement and convergence behavior
- Designing agents that trade off solution quality and computation cost

This project mirrors real-world planning and optimization systems used in routing, scheduling, and resource allocation.
