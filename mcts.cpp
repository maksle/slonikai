#include <iostream>
#include "mcts.h"
#include "position.h"
#include "movegen.h"


bool game_over(const Position& position) {
    return MoveGen<ALL_LEGAL>(position).moves.size() == 0
        || position.arbiter_draw()
        || position.insufficient_material();
}

MCTS::MCTS(string s0, int max_simulations, float c, float w_r, float w_v, float w_a)
    : s0(s0), max_simulations(max_simulations), c(c), w_r(w_r), w_v(w_v), w_a(w_a),
      simulations(0)
{
    if (w_a < 0)
        w_a = max_simulations / 35.0f;
}

Move MCTS::search() {
    while (time_available())
        simulate(Position(s0));
    return select_move(Position(s0), 0);
}

bool MCTS::time_available() {
    return simulations < max_simulations;
}

void MCTS::simulate(Position position) {
    simulations++;
    // cout << ".." << simulations << endl;
    auto states_actions = sim_tree(position);
    float z;
    if (w_r > 0)
        float z = sim_default(Position(get_state(position)));
    else
        float z = 0;
    backup(states_actions, z);
}

vector<tuple<string, Move> > MCTS::sim_tree(Position position) {
    vector<tuple<string, Move> > states_actions;
    while (!game_over(position))
    {
        string s = get_state(position);
        if (!tree.count(s))
        {
            new_node(s);
            states_actions.push_back(make_tuple(s, MOVE_NONE));
            break;
        }
        Move a = select_move(position, this->c);
        states_actions.push_back(make_tuple(s, a));
        position.make_move(a);
    }
    return states_actions;
}

Move MCTS::default_policy(Position position) {
    vector<Move> legal = MoveGen<ALL_LEGAL>(position).moves;
    // vector<Move> legal;
    // generate<ALL_PSEUDO>(pos, moves);
    return * random_choice<>(legal);
}

float MCTS::sim_default(Position position) {
    while (!game_over(position))
    {
        Move a = default_policy(position);
        // cout << a << endl;
        position.make_move(a);
    }
    Side stm = position.side_to_move();
    if (stm == BLACK && position.checkers()) return 1.0;
    else if (stm == WHITE && position.checkers()) return -1.0;
    else return 0.0;
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
    tree[s] = node;

    vector<Move> legal = MoveGen<ALL_LEGAL>(position).moves;
    for (const auto& m : legal)
    {
        MCTSEdge edge;
        edge.Pnn = 0;
        if (usingNN && this->w_a) {
            // int index = move_to_index(m);
            int index = 0;
            edge.Pnn = probs[index];
        }
        edge.Q = 0;
        edge.N = 0;
        tree[s].edges[m] = edge;
    }
}

float MCTS::uct_value(string s, Move a, float c, float w_a, float w_v) {
    MCTSNode node = tree[s];
    MCTSEdge edge = tree[s].edges[a];

    float uct = std::numeric_limits<float>::infinity();
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

float MCTS::lookup_Q(string s, Move a) {
    MCTSEdge edge = tree[s].edges[a];
    return edge.Q;
}

vector<Move> MCTS::pv(string s0) {
    Position position = Position(s0);
    vector<Move> actions;
    string s = s0;
    while (!tree.count(s)) {
        Move a = select_move(position, 0);
        position.make_move(a);
        actions.push_back(a);
        s = get_state(position);
    }
    return actions;
}

Move MCTS::select_move(Position position, float c) {
    string s = get_state(position);
    vector<Move> legal = MoveGen<ALL_LEGAL>(position).moves;

    Move best_move;
    float best_val = NO_EVAL;

    for (const auto& a : legal) {
        float Q = lookup_Q(s, a);
        float uct = uct_value(s, a, c, this->w_a, this->w_v);
        if (best_val == NO_EVAL || (Q + uct) > best_val) {
            best_val = Q + uct;
            best_move = a;
        }
    }

    return best_move;
}

void MCTS::backup(vector<tuple<string, Move> > states_actions, float z) {
    for (const auto& s_a : states_actions) {
        string s = std::get<0>(s_a);
        Move a = std::get<1>(s_a);
        MCTSNode node = tree[s];
        node.N++;
        if (a != MOVE_NONE) {
            MCTSEdge edge = tree[s].edges[a];
            edge.N++;
            // v = (1 - this->w_r) * s_prime_node.Vnn + this->w_r * z;
            float v = z;
            edge.Q += (v - edge.Q) / edge.N;
        }
    }
}

vector<Move> MCTS::get_actions(Position position) {
    return MoveGen<ALL_LEGAL>(position).moves;
}

vector<Move> MCTS::get_actions(string s) {
    return get_actions(Position(s));
}

string MCTS::get_state(const Position& position) {
    return position.fen();
}
