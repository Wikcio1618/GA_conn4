// Population.h

#ifndef BOARD_H
#define BOARD_H

#include <vector>

class Board {
public:
    // ... other class members ...
    Board(int height = 6, int width = 7);
    bool check_winner(std::pair<int, int> move, int player) const;
    bool is_valid_move(int col) const;
    bool is_board_full() const;
    int drop_piece(int col, int player);
    void invert_board();

    int height;
    int width;
    std::vector<std::vector<int>> pieces;
private:
    // ... other private members ...
    bool check_line(const std::vector<int>& line, int player) const;
};

#endif // BOARD_H