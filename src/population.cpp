// Assuming Model and Board classes are defined in separate header files
#include <../include/conn4model.h>
#include <../include/board.h>
#include <../include/population.h>
#include <random>
#include <vector>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <torch/torch.h>

Population::Population(int npop, const std::vector<int> &layer_breaks, double mute_rate)
    : npop(npop), layer_breaks(layer_breaks), mute_rate(mute_rate)
{
    chroms = generateRandomChromosomes();
}

void Population::run_evolution(int max_generations, int num_players_for_tourn)
{
    std::vector<int> player_indexes(npop);
    std::iota(player_indexes.begin(), player_indexes.end(), 0); // Fill with 0, 1, ..., npop-1

    for (int generation = 0; generation < max_generations; ++generation)
    {
        if (generation % 100 == 0)
            std::cout << "Generation " << generation << " is there" << std::endl;
        std::random_shuffle(player_indexes.begin(), player_indexes.end());

        for (size_t i = 0; i < player_indexes.size(); i += num_players_for_tourn)
        {
            std::vector<int> batch(player_indexes.begin() + i, player_indexes.begin() + std::min(i + num_players_for_tourn, player_indexes.size()));
            std::vector<int> tournament_scores = robin_tournament(batch);

            std::vector<std::pair<int, double>> combined_data;
            for (size_t j = 0; j < batch.size(); ++j)
            {
                combined_data.emplace_back(batch[j], tournament_scores[j]);
            }

            std::sort(combined_data.begin(), combined_data.end(), [](const auto &a, const auto &b)
                      { return a.second > b.second; });

            int p1_idx = combined_data[0].first;
            int p2_idx = combined_data[1].first;
            int replaced1_idx = combined_data[combined_data.size() - 1].first;
            int replaced2_idx = combined_data[combined_data.size() - 2].first;

            auto children = crossover(p1_idx, p2_idx);
            replace_worst_individuals(replaced1_idx, replaced2_idx, children);
            apply_mutation(mute_rate, replaced1_idx);
            apply_mutation(mute_rate, replaced2_idx);
        }
    }
}

std::pair<std::vector<double>, std::vector<double>> Population::crossover(int p1_idx, int p2_idx) const
{
    std::vector<double> child1 = chroms[p1_idx];
    std::vector<double> child2 = chroms[p2_idx];

    for (size_t i = 0; i < layer_breaks.size(); ++i)
    {
        int start = (i == 0) ? 0 : layer_breaks[i - 1];
        int end = layer_breaks[i];
        // int c_point = start + 1 + std::rand() % (end - 1);
        int c_point = getRandomInt(start, end);

        for (int j = start; j < c_point; ++j)
        {
            child1[j] = chroms[p2_idx][j];
            child2[j] = chroms[p1_idx][j];
        }
    }

    return std::make_pair(child1, child2);
}

void Population::replace_worst_individuals(int replaced1_idx, int replaced2_idx, const std::pair<std::vector<double>, std::vector<double>> &children)
{
    chroms[replaced1_idx] = children.first;
    chroms[replaced2_idx] = children.second;
}

void Population::apply_mutation(double mute_rate, int idx)
{
    for (size_t i = 0; i < static_cast<size_t>(layer_breaks.back()); ++i)
    {
        if (static_cast<double>(std::rand()) / RAND_MAX < mute_rate)
        {
            chroms[idx][i] += static_cast<double>(std::rand()) / RAND_MAX;
        }
    }
}

int Population::play_game(int p1_idx, int p2_idx)
{
    Board board(6, 7);

    Conn4Model model1(board.width);
    model1.set_parameters(chroms[p1_idx]);

    Conn4Model model2(board.width);
    model2.set_parameters(chroms[p2_idx]);

    int current_player = 1;
    while (true)
    {
        // Get the current player's move from the neural network
        Conn4Model &current_model = (current_player == 1) ? model1 : model2;
        int col = get_model_move(current_model, board);
        // Make the move on the board
        if (board.is_valid_move(col))
        {
            int row = board.drop_piece(col, current_player);
            if (board.check_winner(std::make_pair(row, col), current_player))
            {
                return current_player;
            }
            else if (board.is_board_full())
            {
                return 0;
            }
        }
        else
        {
            // Invalid move, the other player wins
            return 3 - current_player; // Return the winner (1 if 2, 2 if 1)
        }

        current_player = 3 - current_player;
        board.invert_board();
    }
}

int Population::get_model_move(Conn4Model &model, Board &board)
{
    // Convert the board to a PyTorch tensor
    torch::Tensor board_tensor = torch::from_blob(board.pieces.data(), {board.height, board.width}, torch::kDouble);

    // Add batch and channel dimensions
    board_tensor = board_tensor.unsqueeze(0).unsqueeze(0);

    // Put the model in evaluation mode
    model.to(torch::kDouble);
    model.eval();

    // Make a forward pass to obtain the predictions
    torch::NoGradGuard no_grad; // Disable gradient tracking during inference
    torch::Tensor predictions = model.forward(board_tensor);

    // Convert predictions to a NumPy array
    std::vector<double> predictions_vec(predictions.data_ptr<double>(), predictions.data_ptr<double>() + predictions.numel());
    int prediction = static_cast<size_t>(std::distance(predictions_vec.begin(), std::max_element(predictions_vec.begin(), predictions_vec.end())));

    return prediction;
}

std::vector<int> Population::robin_tournament(const std::vector<int> &player_idxs)
{
    std::vector<int> scores(player_idxs.size(), 0);

    for (size_t i = 0; i < player_idxs.size(); ++i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            int result = play_game(player_idxs[i], player_idxs[j]);

            if (result == 0)
            {
                scores[i]++;
                scores[j]++;
            }
            else if (result == 1)
            {
                scores[i] += 3;
            }
            else if (result == 2)
            {
                scores[j] += 3;
            }
        }
    }

    return scores;
}

int Population::cup_tournament()
{
    std::vector<int> player_idxs(npop);
    std::iota(player_idxs.begin(), player_idxs.end(), 0);

    while (player_idxs.size() > 8)
    {
        std::cout << "Beginning new stage of Cup with " << player_idxs.size() << " players left" << std::endl;
        std::shuffle(player_idxs.begin(), player_idxs.end(), std::default_random_engine{});

        // Simulate matches for the current round
        std::vector<int> winners;

        // If the number of players is odd, advance the one without a pair
        if (player_idxs.size() % 2 == 1)
        {
            winners.push_back(player_idxs.back());
            player_idxs.pop_back();
        }

        auto it = player_idxs.begin();
        while (it != player_idxs.end())
        {
            int p1_idx = *it;
            ++it;
            int p2_idx = (it != player_idxs.end()) ? *it : -1; // -1 for cases when the number of players is odd

            // Simulate match (replace with your logic)
            int result;
            if (p2_idx != -1)
            {
                result = play_game(p1_idx, p2_idx);
            }
            else
            {
                // Handle the case when the number of players is odd
                result = 1; // You might want to adjust this based on your game rules
            }

            if (result == 1)
            {
                winners.push_back(p1_idx);
            }
            else if (result == 2)
            {
                winners.push_back(p2_idx);
            }
            else if (result == 0)
            {
                winners.push_back(p1_idx);
                if (p2_idx != -1) // Only push if p2_idx is not -1
                    winners.push_back(p2_idx);
            }

            it++;
        }

        // Update player indexes for the next round
        player_idxs = winners;
    }

    // If there is only one player left, they automatically advance to the next round
    if (player_idxs.size() == 1)
    {
        std::cout << "One player left. Automatically advancing to the next round." << std::endl;
        return player_idxs[0];
    }

    // Otherwise, proceed to the final round-robin tournament
    std::cout << "Final Round Robin Tournament Begins with " << player_idxs.size() << " players" << std::endl;
    std::vector<int> scores = robin_tournament(player_idxs);
    int winner_idx = player_idxs[std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()))];
    std::cout << "The winner is model with index " << winner_idx << std::endl;

    return winner_idx;
}

void Population::save_best(const std::string &path)
{
    int winner_idx = cup_tournament(); // Assuming cup_tournament is a method of your class
    // Conn4Model best_model(width);
    // best_model.set_parameters(chroms[winner_idx]); // Assuming chroms is a member variable

    // Create an output stream to the file
    // torch::serialize::OutputArchive output_archive;
    // best_model.save(output_archive);
    // output_archive.save_to(path);
    // Convert the Conn4Model to a torch::jit::script::Module
    std::ofstream file(path);
    if (file.is_open())
    {
        for (const auto &element : chroms[winner_idx])
        {
            file << element << ",";
        }
        file.close();
        std::cout << "Chromosome saved to " << path << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file for saving chromosome." << std::endl;
    }
}

std::vector<std::vector<double>> Population::generateRandomChromosomes() const
{
    std::vector<std::vector<double>> chromosomes;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < npop; ++i)
    {
        std::vector<double> chromosome;
        for (size_t j = 0; j < static_cast<size_t>(layer_breaks.back()); ++j)
        {
            chromosome.push_back(distribution(gen));
        }
        chromosomes.push_back(chromosome);
    }

    return chromosomes;
}

int Population::getRandomInt(int start, int end) const
{
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define the distribution
    std::uniform_int_distribution<int> distribution(start, end - 1);

    // Generate a random integer
    int randomInt = distribution(gen);

    return randomInt;
}
