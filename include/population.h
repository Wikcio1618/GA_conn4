// Population.h

#ifndef POPULATION_H
#define POPULATION_H

#include <vector>
#include <../include/conn4model.h>
#include <../include/board.h>

class Population {
public:
    // ... other class members ...

    Population(int npop, const std::vector<int>& layer_breaks, double mute_rate);
    void run_evolution(int max_generations, int num_players_for_tourn);
    std::pair<std::vector<double>, std::vector<double>> crossover(int p1_idx, int p2_idx) const;
    void replace_worst_individuals(int replaced1_idx, int replaced2_idx, const std::pair<std::vector<double>, std::vector<double>>& children);
    void apply_mutation(double mute_rate, int idx);
    int play_game(int p1_idx, int p2_idx);
    int get_model_move(Conn4Model& model, Board& board);
    std::vector<int> robin_tournament(const std::vector<int>& player_idxs);
    int cup_tournament();
    void save_best(const std::string& path);
    std::vector<std::vector<double>> generateRandomChromosomes() const;
    int getRandomInt(int start, int end) const;

private:
    // ... other private members ...
    int npop;
    std::vector<int> layer_breaks;
    double mute_rate;
    std::vector<std::vector<double>> chroms;

};

#endif // POPULATION_H
