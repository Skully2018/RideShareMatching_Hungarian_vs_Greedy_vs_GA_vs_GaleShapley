# Ride-Share Matching: Hungarian vs Greedy vs Gale–Shapley vs Genetic Algorithm

## Team Members
- Timothy Chow
- Michael Lamberth

## Project Overview
This project investigates the ride-share matching problem by modeling drivers and riders as a weighted bipartite graph. The goal is to match available drivers with riders as efficiently as possible while comparing the performance of four algorithmic approaches: the Hungarian Algorithm, Greedy Matching, Gale–Shapley Stable Matching, and a Genetic Algorithm. These methods are evaluated based on runtime, number of matches, and overall solution quality.

## Problem Description
In a ride-share system, drivers and riders must be paired in a way that balances efficiency and service quality. Each feasible driver-rider pair is assigned a score based on factors such as pickup distance, wait time, and overall trip quality. This project compares exact, heuristic, stability-based, and evolutionary optimization methods for solving this problem and studies the trade-offs among them.

## Algorithms Implemented
- **Hungarian Algorithm** — an exact weighted bipartite matching method used as a benchmark for solution quality
- **Greedy Matching** — a fast heuristic that repeatedly selects the best available local match
- **Gale–Shapley Stable Matching** — a stability-focused matching method based on ranked preference lists
- **Genetic Algorithm** — a population-based optimization method inspired by natural selection

## Datasets
This project primarily uses synthetic ride-share instances generated to reflect realistic transportation and trip-demand patterns. These datasets allow controlled comparisons across different problem sizes and help evaluate the scalability and effectiveness of each matching approach.

## Repository Structure
- `code/` - algorithm implementations and experiment scripts
- `results/` - raw experiment outputs
- `graphs/` - generated plots and visualizations
- `report/` - final scientific report
- `slides/` - project presentation materials
- `references/` - papers and supporting sources

## How to Run

### Requirements
This project uses Python 3. Install any required dependencies before running the code.

### Clone the Repository
```bash
git clone <repo-url>
cd <repo-name>
```

### Run the Project
```bash
cd code
python main.py
```

### Output
The experiment scripts generate results for comparing the four algorithms, including metrics such as runtime, number of matches, and solution quality. Output files should be saved in the `results/` folder, and any generated graphs should be saved in the `graphs/` folder.

## GenAI Usage Disclosure
Generative AI tools were used to assist with brainstorming, debugging, code organization, and drafting parts of the project documentation. All final implementation decisions, testing, and submitted materials were reviewed and verified by the project team.