# POMDP-Based Temporal Reasoning & Speech Inference

This project implements a probabilistic AI agent capable of reasoning under uncertainty using a Partially Observable Markov Decision Process (POMDP).

## Overview
The agent infers hidden state sequences from noisy observations by modeling uncertainty in both transitions and observations. It is applied to temporal reasoning tasks and speech recognition problems.

## Key Features
- POMDP formulation with:
  - State priors
  - Transition probabilities
  - Observation likelihoods
- Viterbi-based dynamic programming for most-likely state sequence inference
- Supports long-horizon reasoning over sequences of actions and observations
- Applied to:
  - Temporal navigation reasoning
  - Phoneme-to-grapheme mapping for speech recognition

## Technical Details
- Language: Python
- Algorithms: POMDP, Viterbi decoding, probabilistic inference
- Scale:
  - Processes ~300K weighted transitions from large text corpora
- Focus on numerical stability and efficient dynamic programming

## What This Demonstrates
- Probabilistic reasoning under partial observability
- Sequence modeling and inference
- Reliability-focused agent behavior in uncertain environments

This project reflects core techniques used in modern AI systems for speech, language understanding, and decision-making under uncertainty.
