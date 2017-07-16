#include "td.h"
#include "movegen.h"
#include "position.h"
#include "features.h"
#include "search.h"
#include "nn.h"
#include <fstream>
#include <iostream>
#include <string>
#include <cmath> // tanh

namespace {
bool has_moves(const Position& pos) {
    std::vector<Move> moves = MoveGen<ALL_LEGAL>(pos).moves;
    return moves.size() > 0;
}

bool game_over(const Position& pos) {
    return !has_moves(pos) || pos.arbiter_draw();
}
    
} // namespace

namespace TD {

void play() {
    int depth = 4;
    int total_fens = 700762;
    int plies_to_play = 32;
    int positions_per_iteration = 24;
    int num_iterations = total_fens / positions_per_iteration + 1;
    int init_npos = 210225 - 32;
    int offset = init_npos;
    std::vector<int> sts_scores;

    FeatureExtractor extractor;
    
    for (int i = 0; i < num_iterations; ++i)
    {
        std::vector<Position> positions;
        
        std::string filename = "../allfens.txt";
        std::ifstream all_fens(filename);
        
        if (!all_fens) {
            std::cerr << "Failed to open " << filename << " for reading fens" << std::endl;
            assert(false);
        }

        int lines_read = 0;
        std::string line;
        while (lines_read != offset && std::getline(all_fens, line))
            ++lines_read;

        lines_read = 0;
        while (lines_read != positions_per_iteration && std::getline(all_fens, line))
            positions.push_back(Position(line));

        offset += positions_per_iteration;

        int n = offset;
        int m = 0;
        
        for (auto& pos : positions)
        {
            ++n; ++m;
            std::cout << "============================\n";
            std::cout << "New Game, #" << n << ", (#" << m << "/" << positions_per_iteration << "\n";
            std::cout << pos.fen() << "\n";
            std::cout << pos;

            Search::Context context;
            context.root_position = pos;
            context.limits.max_depth = 4;
            context.train.training = true;
            
            while (!game_over(pos)) {
                // clear search_states for treesnap if not using transposition table
                // clear heuristics
                // clear transposition table
                
                SlonikNet net;

                Search::SearchOutput so = Search::iterative_deepening(context);
                Score leaf_val = so.value;
                if (pos.side_to_move() == BLACK)
                    leaf_val = -leaf_val;
                leaf_val /= 10000;

                std::cout << so.pv[0] << " ";
                if (so.pv.size() > 1) {
                    std::cout << "[";
                    for (auto it = so.pv.begin() + 1; it < so.pv.end(); ++i)
                        std::cout << *it << ", ";
                    std::cout << "]";
                }
                
                std::vector< std::vector<std::vector<float>> > features;
                std::vector<float> targets;
                
                for (auto s : context.train.search_states)
                {
                    Position position = Position(s.fen);
                    
                    Score value = s.leaf_eval / 1000;
                    Score static_eval = s.static_eval / 1000;
                    
                    if ((s.bound == LOW_BOUND && value > static_eval)
                        || (s.bound == HIGH_BOUND && value < static_eval)
                        || (s.bound == EXACT_BOUND))
                    {
                        if (position.side_to_move() == BLACK) {
                            value = -value;
                            static_eval = -static_eval;
                        }
                        extractor.set_position(position);
                        std::vector<std::vector<float>> pos_features = extractor.extract();
                        features.push_back(pos_features);
                        targets.push_back(value);
                    }
                }

                // bool is_fixed = eval_is_fixed(leaf, leaf_val)

                net.train(features, targets);

                // if (is_fixed) break;

                pos.make_move(so.pv[0]);

                if (pos.moves.size() % 5 == 0)
                    std::cout << pos;
            } // game 
        } // positions iteration
    } // iterations iteration
}

void initialize(int valid_offset, int valid_num,
                int train_offset, int train_num,
                int batch_size, int valid_frequency)
{
    std::string filename = "../stockfish_init_scores.txt";
    std::ifstream init_file(filename);
        
    if (!init_file) {
        std::cerr << "Failed to open " << filename << " for initialization" << std::endl;
        assert(false);
    }

    FeatureExtractor fe;
    SlonikNet net;
    net.set_batch_size(batch_size);
    
    std::string line;

    int n = 0;
    while (n != valid_offset)
        std::getline(init_file, line);

    n = 0;

    int examples = 0;
    int batches = 0;

    std::vector<Features> valid_features;
    std::vector<float> valid_targets;
    
    std::vector<Features> train_features;
    std::vector<float> train_targets;
    
    while (true)
    {
        std::string fen;
        std::string score_raw;
        std::getline(init_file, fen);
        std::getline(init_file, score_raw);

        float s = std::stoi(score_raw);
        s = std::tanh(s);

        Position pos(fen);
        fe.set_position(pos);
        Features fs = fe.extract();
        
        if (n < valid_num) {
            valid_features.push_back(fs);
            valid_targets.push_back(s);
        }
        else
        {
            train_features.push_back(fs);
            train_targets.push_back(s);

            if (examples == batch_size)
            {
                net.train(train_features, train_targets);

                if (batches > 0 && batches % valid_frequency == 0)
                {
                    float accuracy = net.validate(valid_features, valid_targets);
                    LG << accuracy;
                }

                ++batches;
                train_features.clear();
                train_targets.clear();
                examples = 0;
            }
        }
        
        ++n;

        if (n == (train_offset + train_num))
            break;
    }
}
    
} // namespace TD
