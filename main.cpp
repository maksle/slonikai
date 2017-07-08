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
#include "uci.h"
#include "features.h"
#include "template.cpp"
#include "package/mxnet-cpp/MxNetCpp.h"
#include "nn.h"

#include "sts.h"

// #include "sts.cpp"

int main(int argc, char* argv[])
{
    Bitboards::init();
    Magics::init();
    Zobrist::init();
    TT.resize(256);

    STS::run_sts_test();
    
    // UCI::loop(argc, argv);
    
    // std::cout << "char: " << sizeof(char) << std::endl;
    // std::cout << "int: " << sizeof(int) << std::endl;
    // std::cout << "short int: " << sizeof(short int) << std::endl;
    // std::cout << "TTEntry: " << sizeof(TTEntry) << std::endl;
    
    // Position pos = Position();
    // std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
    // int nodes = Search::perft<true>(pos, 5);
    // std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // std::cout << "Nodes searched: " << nodes << std::endl;

    // Position pos = Position();
    // // nntest(pos);
    // auto net = SlonikNet();
    // // pos.set("r3r1k1/1b1q1pp1/npp2b1p/p2p4/Q2P4/2NBPN2/PP3PPP/2R2RK1 w - - 0 16");
    // std::cout << pos;
    // std::cout << "Value: " << net.evaluate(pos) << std::endl;
    
    // std::cout << "Total ms: " << diff.count() << std::endl;
    // std::cout << "Total ms: " << diff.count() << std::endl;
    // std::cout << "Nodes / second: " << nodes / diff.count() << std::endl;
    
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

    // MXNotifyShutdown();
    return 0;
}
