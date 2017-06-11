#ifndef SEARCH_CPP
#define SEARCH_CPP

#include <vector>
#include <iostream>
#include <string>
#include "types.h"
#include "bb.h"
#include "movegen.h"
#include "position.h"
#include "search.h"

template<bool Root>
int Search::perft(Position& pos, int depth) {
    int cnt = 0;
    int nodes = 0;
    bool leaf = depth == 2;

    std::vector<Move> moves;
    generate<ALL_LEGAL>(pos, moves);

    for (const auto& move : moves)
    {
        if (Root and depth <= 1) {
            cnt = 1;
            nodes++;
        } else {
            std::string fen = pos.fen();
            pos.make_move(move, pos.gives_check(move));
            if (leaf) {
                std::vector<Move> moves2;
                generate<ALL_LEGAL>(pos, moves2);
                cnt = moves2.size();
                // if (from_sq(move) == E3 and to_sq(move) == F4) {
                //     std::cout << pos << std::endl;
                //     std::cout << Bitboards::print_bb(pos.checkers()) << std::endl;
                //     for (auto m : moves2)
                //         std::cout << ":: " << m << " ::" << std::endl;
                // }
            } else {
                cnt = perft<false>(pos, depth - 1);
            }
            pos.unmake_move();
            if (pos.fen() != fen) {
                std::cout << "====\n";
                std::cout << fen << std::endl;
                std::cout << pos.fen() << std::endl;
                for (auto m : pos.moves)
                    std::cout << m << std::endl;
                std::cout << move << std::endl;
                std::cout << "====\n";
                assert(false);
            }
            nodes += cnt;
        }
        if (Root)
            std::cout << move << ": " << cnt << std::endl;
    }
    return nodes;
}

#endif
