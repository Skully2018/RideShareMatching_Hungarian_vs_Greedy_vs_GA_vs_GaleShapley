"""
Ride-Sharing Matching Simulation Framework

This program models and compares multiple matching algorithms for pairing drivers
and riders in a ride-sharing system. The goal is to evaluate tradeoffs between:

- Optimality (Hungarian algorithm)
- Speed (Greedy)
- Stability (Gale-Shapley)
- Heuristic exploration (Genetic Algorithm)

Each algorithm operates on the same generated instance and is evaluated using:
- Total matching score (based on pickup distance)
- Average pickup distance
- Runtime
- Stability violations (for stable matching context)

Design Philosophy:
- Separate instance generation, algorithms, and evaluation
- Ensure fair comparison using a shared scoring function
- Use Hungarian algorithm as a ground-truth benchmark

This structure allows experimentation with algorithmic tradeoffs at scale.
"""

import os
import csv
import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Driver:
    id: int
    location: Point


@dataclass(frozen=True)
class Rider:
    id: int
    pickup: Point


@dataclass(frozen=True)
class Match:
    driver_id: int
    rider_id: int
    distance: float
    score: float


@dataclass
class Instance:
    drivers: List[Driver]
    riders: List[Rider]
    score_matrix: List[List[float]]
    distance_matrix: List[List[float]]
    feasible_matrix: List[List[bool]]
    max_pickup_distance: float


@dataclass
class MatchingResult:
    algorithm: str
    matches: List[Match]
    total_score: float
    average_pickup_distance: float
    runtime_seconds: float
    stability_violations: Optional[int] = None
    metadata: Optional[Dict[str, float]] = None


NEG_INF = -10**15


def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

"""
Scoring function for a driver-rider match.

We use a simple linear decay model:
    score = max(0, 1000 - distance)

Interpretation:
- Shorter pickup distance → higher reward
- Beyond 1000 units → zero reward

This is a simplification of real-world objectives where:
- Time, fuel cost, and rider satisfaction are correlated with distance

NOTE:
Changing this function directly changes the optimization objective
for *all* algorithms.
"""

def compute_score(distance: float) -> float:
    return max(0.0, 1000.0 - distance)


def summarize_matches(
    algorithm: str,
    matches: List[Match],
    runtime_seconds: float,
    stability_violations: Optional[int] = None,
    metadata: Optional[Dict[str, float]] = None,
) -> MatchingResult:
    total_score = sum(m.score for m in matches)
    avg_distance = statistics.mean([m.distance for m in matches]) if matches else 0.0
    return MatchingResult(
        algorithm=algorithm,
        matches=matches,
        total_score=total_score,
        average_pickup_distance=avg_distance,
        runtime_seconds=runtime_seconds,
        stability_violations=stability_violations,
        metadata=metadata or {},
    )


def timed_call(func: Callable, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

"""
Generate a synthetic ride-matching instance.

Drivers and riders are placed randomly in a 2D plane. We precompute:

- distance_matrix: Euclidean distance between each driver-rider pair
- score_matrix: Reward function based on distance (closer = higher score)
- feasible_matrix: Whether a match is allowed (within max pickup distance)

IMPORTANT:
We convert infeasible edges into NEG_INF scores so that:
- Greedy and GA naturally avoid them
- Hungarian can treat them as very costly

This function defines the *entire problem space*, so consistency here
is critical for fair algorithm comparison.
"""

def generate_random_instance(
    num_drivers: int,
    num_riders: int,
    max_pickup_distance: float = 30.0,
    coordinate_limit: float = 100.0,
    seed: Optional[int] = None,
) -> Instance:
    rng = random.Random(seed)

    drivers = [
        Driver(i, Point(rng.uniform(0, coordinate_limit), rng.uniform(0, coordinate_limit)))
        for i in range(num_drivers)
    ]
    riders = [
        Rider(i, Point(rng.uniform(0, coordinate_limit), rng.uniform(0, coordinate_limit)))
        for i in range(num_riders)
    ]

    distance_matrix: List[List[float]] = []
    score_matrix: List[List[float]] = []
    feasible_matrix: List[List[bool]] = []

    for d in drivers:
        dists_row = []
        scores_row = []
        feasible_row = []
        for r in riders:
            dist = euclidean_distance(d.location, r.pickup)
            feasible = dist <= max_pickup_distance
            score = compute_score(dist) if feasible else NEG_INF
            dists_row.append(dist)
            scores_row.append(score)
            feasible_row.append(feasible)
        distance_matrix.append(dists_row)
        score_matrix.append(scores_row)
        feasible_matrix.append(feasible_row)

    return Instance(
        drivers=drivers,
        riders=riders,
        score_matrix=score_matrix,
        distance_matrix=distance_matrix,
        feasible_matrix=feasible_matrix,
        max_pickup_distance=max_pickup_distance,
    )

"""
Greedy matching algorithm.

Approach:
1. Sort all feasible edges by (highest score, then shortest distance)
2. Iteratively select the best available edge
3. Ensure each driver and rider is matched at most once

Properties:
- Fast: O(E log E)
- Simple to implement
- Myopic: makes locally optimal decisions

Limitation:
Greedy does NOT guarantee a globally optimal solution and may perform
significantly worse than Hungarian in dense graphs.
"""

def greedy_matching(instance: Instance) -> List[Match]:
    edges = []
    n = len(instance.drivers)
    m = len(instance.riders)

    for i in range(n):
        for j in range(m):
            if instance.feasible_matrix[i][j]:
                edges.append((instance.score_matrix[i][j], instance.distance_matrix[i][j], i, j))

    edges.sort(key=lambda x: (-x[0], x[1]))

    used_drivers = set()
    used_riders = set()
    matches: List[Match] = []

    for score, distance, i, j in edges:
        if i in used_drivers or j in used_riders:
            continue
        used_drivers.add(i)
        used_riders.add(j)
        matches.append(Match(i, j, distance, score))

    return matches


def build_preferences(instance: Instance) -> Tuple[List[List[int]], List[List[int]], List[Dict[int, int]]]:
    n = len(instance.drivers)
    m = len(instance.riders)

    driver_prefs: List[List[int]] = []
    rider_prefs: List[List[int]] = []
    rider_rank_maps: List[Dict[int, int]] = []

    for i in range(n):
        feasible_riders = [j for j in range(m) if instance.feasible_matrix[i][j]]
        feasible_riders.sort(
            key=lambda j: (-instance.score_matrix[i][j], instance.distance_matrix[i][j], j)
        )
        driver_prefs.append(feasible_riders)

    for j in range(m):
        feasible_drivers = [i for i in range(n) if instance.feasible_matrix[i][j]]
        feasible_drivers.sort(
            key=lambda i: (-instance.score_matrix[i][j], instance.distance_matrix[i][j], i)
        )
        rider_prefs.append(feasible_drivers)
        rider_rank_maps.append({driver_id: rank for rank, driver_id in enumerate(feasible_drivers)})

    return driver_prefs, rider_prefs, rider_rank_maps


"""
Count the number of stability violations in a matching.

A violation occurs when:
- A driver prefers another rider over their assigned one
- AND that rider prefers this driver over their current match

Interpretation:
- 0 violations → stable matching
- Higher values → less stable

This metric allows us to quantify how "unstable" greedy or Hungarian
solutions are compared to Gale-Shapley.
"""

def count_stability_violations(instance: Instance, matches: Sequence[Match]) -> int:
    driver_prefs, _, rider_rank_maps = build_preferences(instance)

    driver_to_rider = {m.driver_id: m.rider_id for m in matches}
    rider_to_driver = {m.rider_id: m.driver_id for m in matches}

    violations = 0
    n = len(instance.drivers)

    for d in range(n):
        current_rider = driver_to_rider.get(d)
        for preferred_rider in driver_prefs[d]:
            if preferred_rider == current_rider:
                break
            current_driver_for_rider = rider_to_driver.get(preferred_rider)
            if current_driver_for_rider is None:
                violations += 1
                continue
            rider_rank_map = rider_rank_maps[preferred_rider]
            if rider_rank_map[d] < rider_rank_map[current_driver_for_rider]:
                violations += 1

    return violations


"""
Gale-Shapley stable matching algorithm (driver-proposing version).

Goal:
Find a stable matching where no driver-rider pair would prefer each other
over their assigned partners.

Key concept:
A "blocking pair" exists if:
- Driver prefers a different rider
- AND that rider prefers that driver over their current match

Properties:
- Guarantees stability
- Not guaranteed to maximize total score
- Produces driver-optimal stable matching

This is included to compare stability vs efficiency tradeoffs.
"""

def gale_shapley_matching(instance: Instance) -> Tuple[List[Match], int]:
    driver_prefs, _, rider_rank_maps = build_preferences(instance)
    n = len(instance.drivers)

    next_proposal_index = [0] * n
    free_drivers = [i for i in range(n) if driver_prefs[i]]
    rider_partner: Dict[int, int] = {}
    driver_partner: Dict[int, int] = {}

    while free_drivers:
        driver = free_drivers.pop(0)
        if next_proposal_index[driver] >= len(driver_prefs[driver]):
            continue

        rider = driver_prefs[driver][next_proposal_index[driver]]
        next_proposal_index[driver] += 1

        if rider not in rider_partner:
            rider_partner[rider] = driver
            driver_partner[driver] = rider
        else:
            current_driver = rider_partner[rider]
            rank_map = rider_rank_maps[rider]
            if rank_map[driver] < rank_map[current_driver]:
                del driver_partner[current_driver]
                free_drivers.append(current_driver)
                rider_partner[rider] = driver
                driver_partner[driver] = rider
            else:
                if next_proposal_index[driver] < len(driver_prefs[driver]):
                    free_drivers.append(driver)

    matches: List[Match] = []
    for driver, rider in sorted(driver_partner.items()):
        matches.append(
            Match(
                driver_id=driver,
                rider_id=rider,
                distance=instance.distance_matrix[driver][rider],
                score=instance.score_matrix[driver][rider],
            )
        )

    violations = count_stability_violations(instance, matches)
    return matches, violations


def hungarian_algorithm(cost: List[List[float]]) -> Tuple[List[int], float]:
    n = len(cost)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        j0 = 0

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0

            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1

    optimal_cost = -v[0]
    return assignment, optimal_cost


"""
Hungarian algorithm for optimal bipartite matching.

Goal:
Maximize total matching score by converting the problem into a
minimum-cost assignment problem.

Approach:
- Convert scores to costs: cost = max_score - score
- Pad matrix to square if needed
- Solve using Hungarian algorithm

Properties:
- Finds globally optimal solution
- Polynomial time: O(n^3)
- Serves as the benchmark for solution quality

Note:
We must carefully handle infeasible edges to avoid invalid matches.
"""

def hungarian_matching(instance: Instance) -> List[Match]:
    n = len(instance.drivers)
    m = len(instance.riders)
    size = max(n, m)

    max_score = 0.0
    for i in range(n):
        for j in range(m):
            if instance.feasible_matrix[i][j]:
                max_score = max(max_score, instance.score_matrix[i][j])

    penalty_cost = max_score + 1_000_000.0
    cost = [[penalty_cost for _ in range(size)] for _ in range(size)]

    for i in range(n):
        for j in range(m):
            if instance.feasible_matrix[i][j]:
                cost[i][j] = max_score - instance.score_matrix[i][j]

    for i in range(size):
        for j in range(size):
            if i >= n or j >= m:
                cost[i][j] = 0.0

    assignment, _ = hungarian_algorithm(cost)

    matches: List[Match] = []
    for i in range(n):
        j = assignment[i]
        if 0 <= j < m and instance.feasible_matrix[i][j]:
            matches.append(
                Match(
                    driver_id=i,
                    rider_id=j,
                    distance=instance.distance_matrix[i][j],
                    score=instance.score_matrix[i][j],
                )
            )

    used_riders = set()
    filtered_matches = []
    for match in matches:
        if match.rider_id not in used_riders:
            filtered_matches.append(match)
            used_riders.add(match.rider_id)

    return filtered_matches


def random_individual(instance: Instance, rng: random.Random) -> List[int]:
    n = len(instance.drivers)
    m = len(instance.riders)
    rider_ids = list(range(m))
    rng.shuffle(rider_ids)

    if m >= n:
        return rider_ids[:n]

    individual = rider_ids[:]
    while len(individual) < n:
        individual.append(-1)
    rng.shuffle(individual)
    return individual


def individual_to_matches(instance: Instance, individual: Sequence[int]) -> List[Match]:
    matches = []
    used_riders = set()

    for i, rider_id in enumerate(individual):
        if rider_id == -1 or rider_id in used_riders:
            continue
        if 0 <= rider_id < len(instance.riders) and instance.feasible_matrix[i][rider_id]:
            matches.append(
                Match(
                    driver_id=i,
                    rider_id=rider_id,
                    distance=instance.distance_matrix[i][rider_id],
                    score=instance.score_matrix[i][rider_id],
                )
            )
            used_riders.add(rider_id)

    return matches


def fitness(instance: Instance, individual: Sequence[int]) -> float:
    total = 0.0
    used_riders = set()

    for i, rider_id in enumerate(individual):
        if rider_id == -1:
            continue
        if rider_id in used_riders:
            total -= 5000.0
            continue
        used_riders.add(rider_id)

        if 0 <= rider_id < len(instance.riders) and instance.feasible_matrix[i][rider_id]:
            total += instance.score_matrix[i][rider_id]
        else:
            total -= 10000.0

    total += 50.0 * len([x for x in individual if x != -1])
    return total


def tournament_selection(
    population: List[List[int]],
    scores: List[float],
    rng: random.Random,
    k: int = 3,
) -> List[int]:
    selected_indices = [rng.randrange(len(population)) for _ in range(k)]
    best_idx = max(selected_indices, key=lambda idx: scores[idx])
    return population[best_idx][:]


def repair_individual(individual: List[int], num_riders: int, rng: random.Random) -> List[int]:
    seen = set()
    duplicates_positions = []

    for idx, rider_id in enumerate(individual):
        if rider_id == -1:
            continue
        if rider_id in seen or rider_id >= num_riders:
            duplicates_positions.append(idx)
        else:
            seen.add(rider_id)

    missing = [r for r in range(num_riders) if r not in seen]
    rng.shuffle(missing)

    for idx in duplicates_positions:
        individual[idx] = missing.pop() if missing else -1

    return individual


def crossover(parent1: List[int], parent2: List[int], num_riders: int, rng: random.Random) -> List[int]:
    n = len(parent1)
    if n <= 1:
        return parent1[:]

    cut1 = rng.randint(0, n - 1)
    cut2 = rng.randint(cut1, n - 1)
    child = parent1[:cut1] + parent2[cut1:cut2] + parent1[cut2:]
    return repair_individual(child, num_riders, rng)


def mutate(individual: List[int], mutation_rate: float, num_riders: int, rng: random.Random) -> List[int]:
    child = individual[:]
    n = len(child)

    for i in range(n):
        if rng.random() < mutation_rate:
            j = rng.randrange(n)
            child[i], child[j] = child[j], child[i]

    if rng.random() < mutation_rate and num_riders > 0:
        i = rng.randrange(n)
        child[i] = rng.randrange(num_riders)

    return repair_individual(child, num_riders, rng)

"""
Genetic Algorithm for approximate matching.

Each individual represents a mapping:
    driver_i → rider_j

Evolution process:
1. Initialize random population
2. Evaluate fitness (total score with penalties)
3. Select parents (tournament selection)
4. Apply crossover and mutation
5. Repair invalid individuals (duplicate riders)
6. Repeat for multiple generations

Properties:
- Heuristic, not guaranteed optimal
- Can explore large search spaces
- Performance depends heavily on fitness design

Key challenge:
Balancing penalties and rewards so that:
- Invalid solutions are discouraged
- Good structures are preserved

This algorithm explores tradeoffs between exploration and exploitation.
"""

def genetic_matching(
    instance: Instance,
    population_size: int = 80,
    generations: int = 120,
    mutation_rate: float = 0.08,
    elite_count: int = 5,
    seed: Optional[int] = None,
) -> Tuple[List[Match], Dict[str, float]]:
    rng = random.Random(seed)
    m = len(instance.riders)

    population = [random_individual(instance, rng) for _ in range(population_size)]
    best_individual = None
    best_score = float("-inf")

    for _ in range(generations):
        scores = [fitness(instance, ind) for ind in population]
        generation_best_idx = max(range(len(population)), key=lambda idx: scores[idx])

        if scores[generation_best_idx] > best_score:
            best_score = scores[generation_best_idx]
            best_individual = population[generation_best_idx][:]

        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        next_population = [ind[:] for _, ind in ranked[:elite_count]]

        while len(next_population) < population_size:
            p1 = tournament_selection(population, scores, rng)
            p2 = tournament_selection(population, scores, rng)
            child = crossover(p1, p2, m, rng)
            child = mutate(child, mutation_rate, m, rng)
            next_population.append(child)

        population = next_population

    if best_individual is None:
        best_individual = random_individual(instance, rng)

    matches = individual_to_matches(instance, best_individual)
    metadata = {
        "ga_population_size": float(population_size),
        "ga_generations": float(generations),
        "ga_mutation_rate": float(mutation_rate),
        "ga_best_fitness": float(best_score),
    }
    return matches, metadata


def run_algorithm(name: str, instance: Instance, ga_params: Optional[Dict] = None) -> MatchingResult:
    if name == "greedy":
        matches, runtime = timed_call(greedy_matching, instance)
        return summarize_matches("greedy", matches, runtime)

    if name == "hungarian":
        matches, runtime = timed_call(hungarian_matching, instance)
        return summarize_matches("hungarian", matches, runtime)

    if name == "gale_shapley":
        (matches, violations), runtime = timed_call(gale_shapley_matching, instance)
        return summarize_matches("gale_shapley", matches, runtime, stability_violations=violations)

    if name == "genetic":
        params = ga_params or {}
        (matches, metadata), runtime = timed_call(genetic_matching, instance, **params)
        return summarize_matches("genetic", matches, runtime, metadata=metadata)

    raise ValueError(f"Unknown algorithm: {name}")


def result_to_row(size: int, trial: int, result: MatchingResult, benchmark_score: Optional[float]) -> Dict[str, float]:
    quality_ratio = (
        result.total_score / benchmark_score
        if benchmark_score is not None and benchmark_score > 0
        else 0.0
    )

    row = {
        "size": size,
        "trial": trial,
        "algorithm": result.algorithm,
        "runtime_seconds": result.runtime_seconds,
        "num_matches": len(result.matches),
        "total_score": result.total_score,
        "average_pickup_distance": result.average_pickup_distance,
        "solution_quality_vs_hungarian": quality_ratio,
        "stability_violations": result.stability_violations if result.stability_violations is not None else "",
    }

    if result.metadata:
        row.update(result.metadata)
    return row

"""
Run experiments across multiple problem sizes and trials.

For each size:
- Generate random instances
- Run all algorithms
- Compare results against Hungarian baseline

Metrics recorded:
- Runtime
- Total score
- Number of matches
- Solution quality vs optimal
- Stability violations

Results are saved to CSV for later analysis and visualization.

This function is the core of the empirical evaluation pipeline.
"""

def run_experiments(
    sizes: Sequence[int] = (50, 100, 250),
    trials_per_size: int = 5,
    max_pickup_distance: float = 30.0,
    algorithms: Sequence[str] = ("hungarian", "greedy", "gale_shapley", "genetic"),
    ga_params: Optional[Dict] = None,
    csv_filename: str = "rideshare_results.csv",
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    for size in sizes:
        for trial in range(1, trials_per_size + 1):
            instance = generate_random_instance(
                num_drivers=size,
                num_riders=size,
                max_pickup_distance=max_pickup_distance,
                seed=None,
            )

            results: Dict[str, MatchingResult] = {}
            for algo in algorithms:
                results[algo] = run_algorithm(algo, instance, ga_params=ga_params)

            benchmark_score = results["hungarian"].total_score if "hungarian" in results else None

            for algo in algorithms:
                rows.append(result_to_row(size, trial, results[algo], benchmark_score))

            print(f"Completed size={size}, trial={trial}")

    if rows:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return rows


def print_single_run_summary(results: Sequence[MatchingResult]) -> None:
    print("\\n=== Single Run Summary ===")
    for result in results:
        print(f"\\nAlgorithm: {result.algorithm}")
        print(f"Runtime: {result.runtime_seconds:.6f} seconds")
        print(f"Matches: {len(result.matches)}")
        print(f"Total score: {result.total_score:.2f}")
        print(f"Average pickup distance: {result.average_pickup_distance:.2f}")
        if result.stability_violations is not None:
            print(f"Stability violations: {result.stability_violations}")
        if result.metadata:
            for key, value in result.metadata.items():
                print(f"{key}: {value}")
        for m in result.matches[:5]:
            print(
                f"  Driver {m.driver_id} -> Rider {m.rider_id} "
                f"| distance={m.distance:.2f} | score={m.score:.2f}"
            )


def print_experiment_summary(rows: Sequence[Dict[str, float]]) -> None:
    print("\\n=== Experiment Summary ===")
    grouped: Dict[Tuple[int, str], List[Dict[str, float]]] = {}
    for row in rows:
        key = (int(row["size"]), str(row["algorithm"]))
        grouped.setdefault(key, []).append(row)

    for (size, algorithm), group_rows in sorted(grouped.items()):
        avg_runtime = statistics.mean(float(r["runtime_seconds"]) for r in group_rows)
        avg_score = statistics.mean(float(r["total_score"]) for r in group_rows)
        avg_matches = statistics.mean(float(r["num_matches"]) for r in group_rows)
        avg_quality = statistics.mean(float(r["solution_quality_vs_hungarian"]) for r in group_rows)
        print(
            f"size={size:4d} | algo={algorithm:12s} | "
            f"avg_runtime={avg_runtime:.6f}s | avg_score={avg_score:.2f} | "
            f"avg_matches={avg_matches:.2f} | quality_vs_hungarian={avg_quality:.4f}"
        )


def main() -> None:
    instance = generate_random_instance(
        num_drivers=10,
        num_riders=10,
        max_pickup_distance=30.0,
        seed=None,
    )

    single_results = [
        run_algorithm("hungarian", instance),
        run_algorithm("greedy", instance),
        run_algorithm("gale_shapley", instance),
        run_algorithm(
            "genetic",
            instance,
            ga_params={
                "population_size": 60,
                "generations": 100,
                "mutation_rate": 0.08,
                "elite_count": 5,
                "seed": None,
            },
        ),
    ]
    print_single_run_summary(single_results)

    results_dir = os.path.join(os.getcwd(), "..", "Results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "rideshare_results.csv")

    rows = run_experiments(
        sizes=(50, 100, 150),
        trials_per_size=3,
        max_pickup_distance=30.0,
        algorithms=("hungarian", "greedy", "gale_shapley", "genetic"),
        ga_params={
            "population_size": 60,
            "generations": 100,
            "mutation_rate": 0.08,
            "elite_count": 5,
            "seed": None,
        },
        csv_filename=csv_path,
    )

    print(f"\\nResults saved to: {csv_path}")
    print_experiment_summary(rows)
    print("\\nSaved experiment results to rideshare_results.csv")


if __name__ == "__main__":
    print("Program started")
    main()
    print("Program finished")