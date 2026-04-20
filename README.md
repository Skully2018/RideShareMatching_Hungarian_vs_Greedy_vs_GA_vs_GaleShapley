# Ride-Share Matching: Hungarian vs Greedy vs Gale–Shapley vs Genetic Algorithm

## Team Members
- Timothy Chow
- Michael Lamberth

## Project Overview
This project studies the ride-share matching problem by modeling drivers and riders as a weighted bipartite graph. We implement and compare the Hungarian Algorithm, Greedy Matching, Gale–Shapley Stable Matching, and a Genetic Algorithm to evaluate trade-offs in runtime, solution quality, and scalability.

## Problem Description
In a ride-share system, available drivers must be matched to riders efficiently. Each feasible driver-rider pair is assigned a score based on factors such as pickup distance, wait time, and overall trip quality. The project compares exact, heuristic, stable, and nature-inspired methods for solving this matching problem.

## Algorithms Implemented
- **Hungarian Algorithm** — exact weighted bipartite matching benchmark
- **Greedy Matching** — fast heuristic based on best available local choice
- **Gale–Shapley Stable Matching** — stability-focused matching using preference lists
- **Genetic Algorithm** — nature-inspired optimization with population-based search

## Datasets
This project primarily uses synthetic ride-share instances generated from realistic transportation patterns. Real-world inspired trip distributions may be incorporated to create more realistic matching scenarios.

## Repository Structure
- `code/` - algorithm implementations and experiment scripts  
- `results/` - raw experiment outputs  
- `graphs/` - generated plots and visualizations  
- `report/` - final scientific report  
- `slides/` - project presentation materials  
- `references/` - papers and supporting sources   

## How to Run

### Clone the repository
```bash
git clone <repo-url>
cd <repo-name>