#include <../include/board.h>

Board::Board(int height, int width) : height(height), width(width)
{
    pieces.resize(height, std::vector<int>(width, 0));
}

bool Board::check_winner(std::pair<int, int> move, int player) const
{
    int row = move.first;
    int col = move.second;

    std::vector<std::vector<int>> lines;
    lines.push_back(pieces[row]);
    std::vector<int> col_line;
    for (int i = 0; i < height; ++i)
    {
        col_line.push_back(pieces[i][col]);
    }
    lines.push_back(col_line);

    std::vector<int> diag1_line, diag2_line;
    for (int i = 0; i < height; ++i)
    {
        int diag1_row = col - row + i;
        int diag2_row = col + row - i;
        if (diag1_row >= 0 && diag1_row < height)
        {
            diag1_line.push_back(pieces[diag1_row][i]);
        }
        if (diag2_row >= 0 && diag2_row < height)
        {
            diag2_line.push_back(pieces[diag2_row][i]);
        }
    }
    lines.push_back(diag1_line);
    lines.push_back(diag2_line);

    for (const auto &line : lines)
    {
        if (check_line(line, player))
        {
            return true;
        }
    }

    return false;
}

bool Board::is_valid_move(int col) const
{
    return pieces[0][col] == 0;
}

bool Board::is_board_full() const
{
    return pieces[0][0] != 0;
}

int Board::drop_piece(int col, int player)
{
    for (int i = height - 1; i >= 0; --i)
    {
        if (pieces[i][col] == 0)
        {
            pieces[i][col] = player;
            return i;
        }
    }

    return -1; // Invalid move (column full)
}

    // Method to invert the board in-place
    void Board::invert_board() {
        for (auto& row : pieces) {
            for (auto& cell : row) {
                if (cell == 1) {
                    cell = 2;  // Change 1 to 2
                } else if (cell == 2) {
                    cell = 1;  // Change 2 to 1
                }
                // Ignore if cell is 0 (empty)
            }
        }
    }

bool Board::check_line(const std::vector<int> &line, int player) const
{
    int count = 0;
    for (int cell : line)
    {
        if (cell == player)
        {
            count++;
            if (count == 4)
            {
                return true;
            }
        }
        else
        {
            count = 0;
        }
    }
    return false;
}
