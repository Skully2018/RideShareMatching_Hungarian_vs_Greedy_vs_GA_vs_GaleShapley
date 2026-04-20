/*
    File name: main.cpp
    Authors: Michael Lamberth and Timothy Chow
    Purpose: This file is the main file to drive the experiment
*/

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

struct Point {
    double x{};
    double y{};
};

struct Driver {
    int id{};
    Point location{};
};

struct Rider {
    int id{};
    Point pickup{};
};

struct Edge {
    int driverId{};
    int riderId{};
    double distance{};
    double score{};
};

struct Match {
    int driverId{};
    int riderId{};
    double distance{};
    double score{};
};

struct MatchingResult {
    std::vector<Match> matches;
    double totalScore{0.0};
    double averagePickupDistance{0.0};
};

double euclideanDistance(const Point& a, const Point& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Higher score = better match.
// For now, use a very simple rule: shorter pickup distance => higher score.
double computeScore(double pickupDistance) {
    return 1000.0 - pickupDistance;
}

std::vector<Edge> buildFeasibleEdges(
    const std::vector<Driver>& drivers,
    const std::vector<Rider>& riders,
    double maxPickupDistance
) {
    std::vector<Edge> edges;

    for (const auto& d : drivers) {
        for (const auto& r : riders) {
            double dist = euclideanDistance(d.location, r.pickup);

            if (dist <= maxPickupDistance) {
                edges.push_back({
                    d.id,
                    r.id,
                    dist,
                    computeScore(dist)
                });
            }
        }
    }

    return edges;
}

MatchingResult greedyMatch(std::vector<Edge> edges) {
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        if (a.score != b.score) return a.score > b.score;   // higher score first
        return a.distance < b.distance;                     // tie-breaker
    });

    std::unordered_set<int> usedDrivers;
    std::unordered_set<int> usedRiders;
    MatchingResult result;

    for (const auto& e : edges) {
        if (usedDrivers.count(e.driverId) == 0 &&
            usedRiders.count(e.riderId) == 0) {

            usedDrivers.insert(e.driverId);
            usedRiders.insert(e.riderId);

            result.matches.push_back({
                e.driverId,
                e.riderId,
                e.distance,
                e.score
            });

            result.totalScore += e.score;
        }
    }

    if (!result.matches.empty()) {
        double totalDistance = 0.0;
        for (const auto& m : result.matches) {
            totalDistance += m.distance;
        }
        result.averagePickupDistance = totalDistance / result.matches.size();
    }

    return result;
}

// Simple synthetic instance generator for testing.
std::vector<Driver> generateDrivers(int n, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> coordDist(0.0, 100.0);

    std::vector<Driver> drivers;
    drivers.reserve(n);

    for (int i = 0; i < n; ++i) {
        drivers.push_back({i, {coordDist(rng), coordDist(rng)}});
    }

    return drivers;
}

std::vector<Rider> generateRiders(int n, int seed = 1337) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> coordDist(0.0, 100.0);

    std::vector<Rider> riders;
    riders.reserve(n);

    for (int i = 0; i < n; ++i) {
        riders.push_back({i, {coordDist(rng), coordDist(rng)}});
    }

    return riders;
}

void printResult(const MatchingResult& result) {
    std::cout << "Matches found: " << result.matches.size() << "\n";
    std::cout << "Total score: " << std::fixed << std::setprecision(2)
              << result.totalScore << "\n";
    std::cout << "Average pickup distance: "
              << result.averagePickupDistance << "\n\n";

    std::cout << "Assignments:\n";
    for (const auto& m : result.matches) {
        std::cout << "  Driver " << m.driverId
                  << " -> Rider " << m.riderId
                  << " | distance = " << m.distance
                  << " | score = " << m.score << "\n";
    }
}

int main() {
    const int numDrivers = 8;
    const int numRiders = 8;
    const double maxPickupDistance = 30.0;

    auto drivers = generateDrivers(numDrivers);
    auto riders = generateRiders(numRiders);

    auto edges = buildFeasibleEdges(drivers, riders, maxPickupDistance);

    std::cout << "Drivers: " << drivers.size() << "\n";
    std::cout << "Riders: " << riders.size() << "\n";
    std::cout << "Feasible edges: " << edges.size() << "\n\n";

    auto result = greedyMatch(edges);
    printResult(result);

    return 0;
}