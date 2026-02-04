# Little-Go AI Agent

This project implements an autonomous AI agent for the 5×5 Go game (Little-Go), designed to reason, plan, and execute multi-step decisions under strict game constraints.

## Overview
The agent models the Go environment and selects optimal moves using adversarial search and heuristic evaluation. It is designed to operate efficiently under time and move limits while competing against multiple baseline and adversarial agents.

## Key Features
- Minimax search with alpha-beta pruning for efficient decision-making
- Rule-aware move generation handling liberties, captures, KO rule, and no-suicide constraints
- Custom heuristic evaluation incorporating:
  - Stone advantage
  - Liberty counts
  - Board topology (Euler characteristic)
  - Edge penalties and komi compensation
- Designed for reliable performance under constrained branching and depth limits

## Technical Details
- Language: Python
- Algorithms: Minimax, Alpha-Beta Pruning
- Environment: Deterministic, adversarial, turn-based
- Evaluation: Tested across 80+ simulated matches against baseline AI agents

## What This Demonstrates
- Agentic reasoning and long-horizon planning
- Performance optimization through search-space pruning
- Designing reliable decision systems under constraints

This project reflects core principles used in real-world AI agents that must reason, act, and adapt in complex environments.
