#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <map>
#include <tuple>

inline std::vector<std::string> split_str(std::string src, std::string delim)
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

inline std::string trim(std::string src) {
    return src.substr(src.find_first_not_of(' '), src.find_last_not_of(' ') - 1);
}

inline std::tuple<std::string,
                  std::vector<std::string>,
                  std::map<std::string, int>>
parse_epd(std::string epd) {
    auto ops = split_str(epd, ";");
    auto fen_parts = split_str(ops[0], " ");
    std::ostringstream fen_os;
    for (auto& s : fen_parts)
        fen_os << s;
    fen_os << " 0 0";
    std::string fen = fen_os.str();

    std::cout << "a" << std::endl;
    
    std::string scores_str, moves_str;
    
    for (auto it = ops.begin() + 1; it < ops.end(); ++it)
    {
        std::string op = trim(*it);
        if (op.find("c8") != std::string::npos)
            scores_str = op;
        else if (op.find("c9") != std::string::npos)
            moves_str = op;
    }
    
    std::cout << "b" << std::endl;
    
    std::string scores = split_str(scores_str, "\"")[1];
    std::vector<int> int_scores;
    for (const auto& s : split_str(scores, " "))
        int_scores.push_back(atoi(s.c_str()));
    auto uci_moves = split_str(split_str(moves_str, "\"")[1], " ");

    std::cout << "c" << std::endl;
    
    std::map<std::string, int> move_score;
    for (size_t i = 0; i < uci_moves.size(); ++i) {
        move_score[uci_moves[i]] = int_scores[i];
    }

    std::cout << "d" << std::endl;

    std::tuple<std::string, std::vector<std::string>, std::map<std::string, int>> res;
    std::cout << "e" << std::endl;
    std::get<0>(res) = fen;
    std::cout << "f" << std::endl;
    std::get<1>(res) = uci_moves;
    std::cout << "g" << std::endl;
    std::get<2>(res) = move_score;
    std::cout << "h" << std::endl;
    
    return res;
}
