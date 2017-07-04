#include <string>
#include <iostream>
#include <cmath> // pow
#include "uci.h"
#include "types.h"
#include "position.h"
#include "search.h"
#include "constants.h"

namespace {
    std::string start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
}

void UCI::loop(int argc, char* argv[]) {

    Position pos;
    pos.set(start_fen);

    std::string token, cmd;
    
    Search::SearchInfo si[256];

    Search::Context context;
    context.root_position = pos;
    
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << "depth " << i << std::endl;
    //     si->pv.clear();
    //     Search::search<true>(pos, si, NEGATIVE_INF, POSITIVE_INF, pow(4, i));
    //     for (auto &m : si->pv) {
    //         std::cout << m << std::endl;
    //     }
    //     std::cout << "\n";
    // }
}

std::string UCI::str(Move m)
{
    std::stringstream ss;
    Square from = from_sq(m);
    Square to = to_sq(m);
    ss << std::string { char(file_of(from) + 'a'), char(rank_of(from) + '1') };
    ss << std::string { char(file_of(to) + 'a'), char(rank_of(to) + '1') };
    if (type_of(m) == PROMOTION)
        ss << PieceToChar[promo_piece(m)];
}
