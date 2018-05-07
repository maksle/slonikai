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

                net.fit(features, targets);

                // if (is_fixed) break;

                pos.make_move(so.pv[0]);

                if (pos.moves.size() % 5 == 0)
                    std::cout << pos;
            } // game 
        } // positions iteration
    } // iterations iteration
}

void initialize()
{
    int batch_size = 32;
    int train_num = 235705;
    int valid_frequency = 16384;
    int valid_num = 60000;

    FeatureExtractor fe;
    SlonikNet net;
    // return;
    // net.set_batch_size(batch_size);
    
    // std::string line;

    int n = 0;
    // while (n != valid_offset)
    //     std::getline(fens_stream, line);
    //     std::getline(scores_stream, line);

    n = 0;

    int examples = 0;
    int batches = 0;

    std::vector<Features> valid_features;
    std::vector<float> valid_targets;
    
    std::vector<Features> train_features;
    std::vector<float> train_targets;
    
    int epochs = 11;
    
    std::string data_path = "/home/maksle/share/slonik_data/shuffled/";
    // std::string data_path = "/home/maks/projects/slonik_data/shuffled/";

    std::ifstream fens_valid_stream(data_path + "fen_valid.txt");
    std::ifstream scores_valid_stream(data_path + "score_valid.txt");
    
    if (!fens_valid_stream) {
        std::cerr << "Failed to open fens validation file" << std::endl;
        assert(false);
    }

    if (!scores_valid_stream) {
        std::cerr << "Failed to open scores validation for initialization" << std::endl;
        assert(false);
    }
    
    for (int v = 0; v < valid_num; v++) {
        std::string fen;
        std::string score;
        std::getline(fens_valid_stream, fen);
        std::getline(scores_valid_stream, score);
        float s = std::stof(score);
        
        Position pos(fen);
        fe.set_position(pos);
        Features fs = fe.extract();
        
        valid_features.push_back(fs);
        valid_targets.push_back(s);
    }
    
    float accuracy = net.validate(valid_features, valid_targets);
    std::cout << accuracy;
    
    for (int i = 0; i < epochs; i++) {
        // float accuracy = net.validate(valid_features, valid_targets);
        // LG << accuracy;

        std::stringstream ss;
        std::string fens_fname;
        ss << data_path << "fen" << i << ".txt";
        ss >> fens_fname;
        std::ifstream fens_stream(fens_fname);
        if (!fens_stream) {
            std::cerr << "Failed to open fens file" << fens_fname << std::endl;
            assert(false);
        }

        ss.clear();
        ss.str(std::string());
        std::string scores_fname;
        ss << data_path << "score" << i << ".txt";
        ss >> scores_fname;
        std::ifstream scores_stream(scores_fname);
        if (!scores_stream) {
            std::cerr << "Failed to open scores file" << scores_fname << std::endl;
            assert(false);
        }
        
        for (int k = 0; k < train_num; k++)
        {
            std::string fen;
            std::string score;
            std::getline(fens_stream, fen);
            std::getline(scores_stream, score);
            float s = std::stof(score);
            
            Position pos(fen);
            fe.set_position(pos);
            Features fs = fe.extract();
            
            train_features.push_back(fs);
            train_targets.push_back(s);
            examples++;
            
            if (examples == batch_size)
            {
                net.fit(train_features, train_targets);
                // exit(0);

                if (batches > 0 && batches % valid_frequency == 0)
                {
                    float accuracy = net.validate(valid_features, valid_targets);
                    std::cout << accuracy;
                }

                ++batches;
                train_features.clear();
                train_targets.clear();
                examples = 0;
            }
        }
        float accuracy = net.validate(valid_features, valid_targets);
        std::cout << accuracy;

        std::stringstream checkpoint_name;
        checkpoint_name << "epoch_" << i << "_vloss_" << accuracy << ".params";
        net.save(checkpoint_name.str());
    }
}
    
} // namespace TD
