#include <../include/population.h> // Include the header for your Population class
#include <../include/conn4model.h> // Include the header for your ConnectFourNN class
#include <../include/board.h>
#include <fstream>

int main()
{
    // torch::set_num_threads(1);         // Set the number of threads
    // torch::set_num_interop_threads(1); // Set the number of interop threads
    // torch::set_num_torch_threads(1);   // Set the number of TorchScript interpreter threads

    const int npop = 1000;
    const int num_gens = 1000;
    const int num_checkpoints = 8;
    const double mute_rate = 0.02;
    const int width = 7;

    // Get the layer sizes for ConnectFourNN
    Conn4Model sample_nn(width);
    std::vector<size_t> layer_sizes = sample_nn.get_weight_breakpoints();

    // Calculate layer breaks
    std::vector<int> layer_breaks;
    int cur = 0;
    for (int num : layer_sizes)
    {
        cur += num;
        layer_breaks.push_back(cur);
    }

    // Create Population
    Population pop(npop, layer_breaks, mute_rate);

    // Write parameters to file
    std::ofstream file("../05.01.23_weekend/params.txt");
    if (file.is_open())
    {
        file << "npop = " << npop << "\n";
        file << "num_gens = " << num_gens << "\n";
        file << "mute_rate = " << mute_rate << "\n";
        file.close();
    }
    else
    {
        std::cerr << "Error opening params.txt for writing." << std::endl;
    }

    // Run evolution and save checkpoints
    for (int checkpoint = 1; checkpoint <= num_checkpoints; ++checkpoint)
    {
        pop.run_evolution(num_gens / num_checkpoints, 5);
        try
        {
            // Save the best_model to the specified file path
            pop.save_best("check" + std::to_string(checkpoint) + ".txt");
            std::cout << "Model saved successfully." << std::endl;
        }
        catch (const c10::Error &e)
        {
            std::cerr << "Error saving the model: " << e.msg() << std::endl;
        }
    }

    return 0; // Return success code
}