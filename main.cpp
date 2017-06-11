#include <vector>
#include <iostream>
#include <chrono>
#include "bb.h"
#include "zobrist.h"
#include "magics.h"
#include "movegen.h"
#include <bitset>
#include "tt.h"
#include "position.h"
#include "search.h"

typedef uint64_t ULL;

int main() {
    Bitboards::init();
    Magics::init();
    Zobrist::init();
    
    // const int CacheLineSize = 64;
    // TTEntry me[1] {};
    // me[0] = TTEntry();
    
    Position pos = Position();
    // pos.make_move(make_move(F1, F2), pos.gives_check(make_move(F1, F2)));
    // pos.make_move(make_move<PROMOTION>(B2, A1, QUEEN), pos.gives_check(make_move<PROMOTION>(B2, A1, QUEEN)));
    
    // std::cout << pos << std::endl;
    // std::cout << pos.fen() << std::endl;
    // std::cout << pos.en_pessant_sq() << std::endl;
    // std::cout << "checkers: " << popcount(pos.checkers()) << std::endl;
    
    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

    int nodes = Search::perft<true>(pos, 5);

    std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();

    // std::chrono::milliseconds diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::chrono::duration<double> diff = end - start;

    std::cout << "Nodes searched: " << nodes << std::endl;
    std::cout << "Total ms: " << diff.count() << std::endl;
    std::cout << "Total ms: " << diff.count() << std::endl;
    std::cout << "Nodes / second: " << nodes / diff.count() << std::endl;
    
    // std::cout << Bitboards::print_bb(pos.discoverers(BLACK)) << std::endl;
    
    // std::vector<Move> moves;
    // generate<ALL_PSEUDO>(pos, moves);
    // // generate<ALL_LEGAL>(pos, moves);
    // std::cout << moves.size() << std::endl;
    // for (auto move : moves) {
    //     Square from = from_sq(move);
    //     Square to = to_sq(move);
    //     Piece pt = pos.piece_on(from);
    //     std::cout << PicturePieces[static_cast<int>(pt)] << ' '
    //               << move
    //               << " legal: " << pos.is_legal(move)
    //               << std::endl;
    // }
    
    return 0;
}
