#include <string>
#include <vector>
#include <cassert>
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <tuple>
#include "sts.h"
#include "position.h"
#include "types.h"
#include "search.h"
#include "uci.h"

namespace {
    std::vector<std::string> split_str(std::string src, std::string delim)
    {
        std::vector<std::string> res;
        size_t start = 0;
        size_t pos = src.find(delim);
        while (pos != std::string::npos) {
            std::string token = src.substr(start, pos - start);
            res.push_back(token);
            start = pos + delim.length();
            pos = src.find(delim, start);
        }
        if (start < src.length())
            res.push_back(src.substr(start, src.length() - start));
        return res;
    }

    std::string trim(std::string src)
    {
        return src.substr(src.find_first_not_of(' '), src.find_last_not_of(' ') - 1);
    }

    std::tuple<std::string, std::map<std::string, int>>
        parse_epd(std::string epd)
    {
        // Quick and dirty epd parser just good enough to get through the 1500 STS EPDs
        auto ops = split_str(epd, ";");
        auto fen_parts = split_str(ops[0], " ");
        std::ostringstream fen_os;
        for (size_t i = 0; i < 4; ++i) {
            fen_os << fen_parts[i] << " ";
        }
        fen_os << " 0 0";
        std::string fen = fen_os.str();

        std::string scores_str, moves_str;

        for (auto it = ops.begin() + 1; it < ops.end(); ++it)
        {
            std::string op = trim(*it);
            if (op.find("c8") != std::string::npos)
                scores_str = op;
            else if (op.find("c9") != std::string::npos)
                moves_str = op;
        }

        std::string scores = split_str(scores_str, "\"")[1];
        std::vector<int> int_scores;
        for (const auto& s : split_str(scores, " "))
            int_scores.push_back(atoi(s.c_str()));
        auto uci_moves = split_str(split_str(moves_str, "\"")[1], " ");

        std::map<std::string, int> move_score;
        for (size_t i = 0; i < uci_moves.size(); ++i) {
            move_score[uci_moves[i]] = int_scores[i];
        }

        std::tuple<std::string, std::map<std::string, int>> res;
        std::get<0>(res) = fen;
        std::get<1>(res) = move_score;

        return res;
    }
}

namespace STS {

int run_sts_test()
{
    // std::map<int, int> total_scores;
    // std::map<int, int> best_counts;
    int n = 0;
    int score = 0;

    std::string filename = "./tools/STS1-STS15.EPD";
    std::ifstream sts_file(filename);

    if (!sts_file) {
        std::cerr << "Failed to open " << filename << " for reading" << std::endl;
        assert(false);
    }

    n = 0;
    std::string line;
    while (std::getline(sts_file, line))
    {
        n += 1;
        std::string fen;
        std::map<std::string, int> move_scores;
        std::tie(fen, move_scores) = parse_epd(line);

        Position pos(fen);
        std::cout << pos << std::endl;

        Search::Context context;
        context.root_position = pos;
        context.limits.max_depth = 2;
        Search::SearchOutput so = Search::iterative_deepening(context);

        if (so.pv.size() == 0)
            std::cerr << "No result from search" << std::endl;

        std::cout << "Chosen move: " << UCI::str(so.pv[0]) << std::endl;
        auto it = move_scores.find(UCI::str(so.pv[0]));
        if (it != move_scores.end()) {
            score += it->second;
            std::cout << it->second << std::endl;
        }
    }

    std::cout << "Final score: " << score;
    return score;
}
    
}
