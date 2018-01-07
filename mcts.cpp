#include <iostream>
#include "mcts.h"
#include "position.h"
#include "bb.h"


enum GameTerminationReason {
    NONE, NO_MOVES, INSUFFICIENT_MATERIAL, NO_PROGRESS
};

bool game_over(const Position& position, GameTerminationReason& reason) {
    if (MoveGen<ALL_LEGAL>(position).moves.size() == 0) {
        reason = NO_MOVES;
        return true;
    }
    if (position.arbiter_draw()) {
        reason = NO_PROGRESS;
        return true;
    }
    if (position.insufficient_material()) {
        reason = INSUFFICIENT_MATERIAL;
        return true;
    }
    return false;
}

MCTS::MCTS(string s0, int max_simulations, float c, float w_r, float w_v, float w_a)
    : s0(s0), max_simulations(max_simulations), c(c), w_r(w_r), w_v(w_v), w_a(w_a), simulations(0)
{
    if (w_a < 0)
        w_a = max_simulations / 35.0f;
}

Move MCTS::search() {
    Position root = Position(s0);
    while (time_available())
        simulate(root);
    return select_move(root, 0);
}

bool MCTS::time_available() {
    return simulations < max_simulations;
}

void MCTS::simulate(Position position) {
    simulations++;
    auto states_actions = sim_tree(position);
    float z;
    if (w_r > 0)
        z = sim_default(position);
    else
        z = 0;
    backup(states_actions, z);
}

vector<tuple<string, Move> > MCTS::sim_tree(Position& position) {
    vector<tuple<string, Move> > states_actions;
    string s = get_state(position);
    string prev_s(s); 
    GameTerminationReason reason;
    while (!game_over(position, reason))
    {
        s = get_state(position);
        if (!tree.count(s))
        {
            if (prev_s == s || tree.at(prev_s).N > 1) {
                new_node(s);
                states_actions.push_back(make_tuple(s, MOVE_NONE));
            }
            break;
        }
        Move a = select_move(position, this->c);
        states_actions.push_back(make_tuple(s, a));
        position.make_move(a);
        prev_s = s;
    }
    return states_actions;
}

Move MCTS::default_policy(const Position& position) {
    vector<Move> legal = MoveGen<ALL_LEGAL>(position).moves;
    return *(random_choice.select<>(legal));
}

float MCTS::sim_default(Position position) {
    GameTerminationReason reason;
    while (!game_over(position, reason))
    {
        Move a = default_policy(position);
        position.make_move(a);
    }

    float result = 0.0;
    Side stm = position.side_to_move();
    if (reason == NO_MOVES && position.checkers())
        if (stm == BLACK) // black is checkmated
            result = 1.0;
        else
            result -1.0;

    return result;
}

void MCTS::new_node(string s) {
    Position position = Position(s);

    bool usingNN = this->w_a || this->w_v;

    PositionEvaluation evals;
    float v;
    vector<float> probs;
    if (usingNN) {
        evals = evaluator(position);
        v = std::get<0>(evals);
        probs = std::get<1>(evals);
    }

    MCTSNode node;
    node.N = 0;
    node.Vnn = NO_EVAL;
    if (usingNN) {
        node.Vnn = v;
    }
    tree.emplace(std::make_pair(s, std::move(node)));

    vector<Move> legal = get_actions(position);
    for (const auto& m : legal)
    {
        MCTSNode edge;
        edge.Pnn = 0;
        if (usingNN && this->w_a) {
            // int index = move_to_index(m);
            int index = 0;
            edge.Pnn = probs[index];
        }
        edge.Q = 0;
        edge.N = 0;
        tree.emplace(std::make_pair(edge_key(s, m), std::move(edge)));
        // tree[edge_key(s, m)] = edge;
    }
}

string MCTS::edge_key(string s, Move a) const {
    Square from = from_sq(a);
    Square to = to_sq(a);
    MoveType mt = type_of(a);
    
    string res = s;
    res += char(file_of(from) + 'a');
    res += char(rank_of(from) + '1');
    res += char(file_of(to) + 'a');
    res += char(rank_of(to) + '1');
    if (mt == PROMOTION) {
        PieceType bt = promo_piece(a);
        if (bt != QUEEN && bt != PIECETYPE_NONE) {
            res += PieceToChar[bt + 6 /*lowercase*/];
        }
    }
    return res;
}

float MCTS::lookup_Q(string s, Move a) const {
    return (tree.find(edge_key(s, a))->second).Q;
}

float MCTS::uct_value(string s, Move a, float c, float w_a, float w_v) const {
    MCTSNode node = tree.at(s);
    MCTSNode edge = tree.at(edge_key(s, a));

    float uct = 0;
    if (edge.N > 0)
        uct = c * sqrt(log(node.N) / edge.N);

    float policy_prior_bonus = 0;
    if (w_a)
        policy_prior_bonus = w_a * edge.Pnn / (edge.N + 1);

    float value_prior_bonus = 0;
    // if (w_v && s_prime_node.Vnn != NO_EVAL)
    // if (w_v)
    //     value_prior_bonus = w_v * s_prime_node.Vnn;

    return uct + policy_prior_bonus + value_prior_bonus;
}

vector<Move> MCTS::pv(string s0) const {
    Position position = Position(s0);
    vector<Move> actions;
    string s = s0;
    while (tree.count(s)) {
        Move a = select_move(position, 0);
        position.make_move(a);
        actions.push_back(a);
        s = get_state(position);
    }
    return actions;
}

Move MCTS::select_move(const Position& position, float c) const {
    string s = get_state(position);
    vector<Move> legal = get_actions(position);

    Move best_move;
    float best_val = NO_EVAL;

    Side stm = position.side_to_move();
    
    for (const auto& a : legal) {
        float Q = lookup_Q(s, a);
        float uct = uct_value(s, a, c, this->w_a, this->w_v);
        float QU = Q + uct;
        if (stm == BLACK)
            QU = -QU;
        if (best_val == NO_EVAL || QU > best_val)
        {
            best_val = Q + uct;
            best_move = a;
        }
    }

    return best_move;
}

void MCTS::backup(const vector<tuple<string, Move> >& states_actions, float z) {
    for (const auto& s_a : states_actions) {
        string s = std::get<0>(s_a);
        Move a = std::get<1>(s_a);
        MCTSNode& node = tree.at(s);
        node.N++;
        if (a != MOVE_NONE) {
            MCTSNode& edge = tree.at(edge_key(s, a));
            edge.N++;
            // v = (1 - this->w_r) * s_prime_node.Vnn + this->w_r * z;
            float v = z;
            edge.Q += (v - edge.Q) / edge.N;
        }
    }
}

vector<Move> MCTS::get_actions(const Position& position) const {
    return MoveGen<ALL_LEGAL>(position).moves;
}

vector<Move> MCTS::get_actions(string s) const {
    return get_actions(Position(s));
}

string MCTS::get_state(const Position& position) const {
    return position.fen();
}
