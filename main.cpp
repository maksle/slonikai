#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
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
#include "td.h"
#include "sts.h"
// #include "sts.cpp"
#include "mcts.h"

int main(int argc, char* argv[])
{
    Bitboards::init();
    Magics::init();
    Zobrist::init();
    TT.resize(256);

    string sims_ = argv[1];
    int sims = stoi(sims_);
    
    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
    
    std::string s0 = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";
    auto mcts = MCTS(s0, sims, sqrt(2), 1.0, 0.0, 0.0);
    Move move = mcts.search();
    std::cout << move << std::endl;
    for (const auto& m : mcts.pv(s0)) {
        std::cout << m << " ";
    }
    std::cout << std::endl;

    std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    std::cout << "secs: " << diff.count() << std::endl;
    
    // TD::initialize();

    // SlonikNet net;

    // auto args_map = net.executor->arg_dict();
    // for (const auto& kv : args_map) {
    //     LG << kv.first;
    //     LG << kv.second;
    //     LG << " ";
    // }
    
    // float accuracy = net.validate(valid_features, valid_targets);
    // LG << accuracy;
    
    // std::string name = "test.params";
    // net.save(name);

    // delete net;

    // Sloniknet net2;
    

    // SlonikNet net;

    // std::string filename= "../outtest.txt";
    // std::ifstream fstream(filename);
    // std::string val;
    // std::streampos pos;
    // for (int i = 0; i < 6; ++i) {
    //     std::getline(fstream, val);
    //     if (i == 4) {
    //         pos = fstream.tellg();
    //         std::cout << "pos at 4 is " << pos << ", val is " << val << std::endl;
    //     }
    //     std::cout << fstream.good() << std::endl;
    // }
    // fstream.seekg(pos);
    // std::cout << "set pos to " << pos << std::endl;
    // std::cout << "stream is good?: " << fstream.good() << std::endl;
    // std::getline(fstream, val);
    // std::cout << val << std::endl;
    
    // std::string filename = "/home/maksle/share/slonik_data/stockfish_init_scores.txt";
    // std::string fen_fn = "/home/maksle/share/slonik_data/stockfish_init_fens.txt";
    // std::string score_raw_fn = "/home/maksle/share/slonik_data/stockfish_init_scores_raw.txt";
    // std::string score_std_fn = "/home/maksle/share/slonik_data/stockfish_init_scores_std.txt";
    // // std::string filename = "/home/maksle/share/sts_scores.txt";
    // std::ifstream train_stream(filename);
    // std::ofstream fen_stream(fen_fn);
    // std::ofstream score_raw_stream(score_raw_fn);
    // std::ofstream score_std_stream(score_std_fn);
    // // std::ifstream train_stream("../outtest.txt");
    // // std::ofstream out_stream("../outtest.txt");
    // int n = 0;
    // std::string fen;
    // std::string score;
    
    // float min = -6.604f;
    // float max = 6.304f;
    // float mean = 0.0271595f;
    // float std = 0.524705f;
    // // float min = 10;
    // // float max = -10;
    // float sum = 0;
    
    // while (train_stream) {
    //     std::getline(train_stream, fen);
    //     std::getline(train_stream, score);
    //     if (!train_stream) break;
        
    //     float s = std::stoi(score)/1000.0f;
    //     // float s = std::stoi(score);
    //     n++;
    //     // sum += s;
    //     // if (s < min) min = s;
    //     // if (s > max) max = s;
        
    //     // s = (s - mean) / std;
    //     // s = (2.0f * (s - min) / (max - min)) - 1;
    //     s = tanh(s / std);
        
    //     fen_stream << fen << std::endl;
    //     score_raw_stream << score << std::endl;
    //     score_std_stream << s << std::endl;
    // }
    
    // std::cout << "sum " << sum << " n " << n << std::endl;
    // std::cout << "min " << min << " max " << max << std::endl;
    // std::cout << "avg " << sum / float(n) << std::endl;
    
    // mean: 0.0271595, min: -6.604, max: 6.304
    // std: sqrt(average((x - x.mean())**2))
    // normalize: (2 * (x - x.min()) / (x.max() - x.min())) - 1
    // stdize: (x - x.mean()) / x.std()
    
    // std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
    // STS::run_sts_test();
    // std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
    
    // std::chrono::duration<double> diff = end - start;
    // std::cout << "secs: " << diff.count() << std::endl;
    
    // UCI::loop(argc, argv);
    
    // std::cout << "char: " << sizeof(char) << std::endl;
    // std::cout << "int: " << sizeof(int) << std::endl;
    // std::cout << "short int: " << sizeof(short int) << std::endl;
    // std::cout << "TTEntry: " << sizeof(TTEntry) << std::endl;
    
    // int depth = 5;
    // Position pos = Position("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
    // Position pos = Position("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 0");
    // Position pos;
    // std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
    // int nodes = Search::perft<true>(pos, depth);
    // std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // std::cout << "Depth: " << depth << std::endl;
    // std::cout << "Seconds: " << diff.count() << std::endl;
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
