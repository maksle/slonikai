#ifndef MCTS_H_GUARD
#define MCTS_H_GUARD

#include <vector>
#include <unordered_map>
#include <string>
#include <functional>
#include <cmath> // sqrt, log
#include <random>
#include "types.h"
#include "movegen.h"

using namespace std;

class Position;

const int NO_EVAL = 10;

struct MCTSNode {
  // Nodes
  int N = 0;
  float Vnn = 0;
  // Edges
  int Q = 0;
  float Pnn = 0;
};

class RandomSelector
{
 private:
  std::mt19937 _mt;

 public:
 RandomSelector() {
   std::random_device rd;
   _mt = std::mt19937(rd());
 }
  template<typename T> typename vector<T>::const_iterator
    select (const vector<T>& choices) {
    std::uniform_int_distribution<> dis(0, choices.size() - 1);
    auto it = choices.begin();
    std::advance(it, dis(_mt));
    return it;
  }
};

typedef tuple<float, vector<float>> PositionEvaluation;
typedef std::function<PositionEvaluation(const Position&)> MCTSEvaluator;

class MCTS {
 private:
  string s0;
  float c;
  float w_r;
  float w_a;
  float w_v;
  MCTSEvaluator evaluator;
  unordered_map<string, MCTSNode> tree;
  int simulations;
  int max_simulations;
  RandomSelector random_choice;

 public:

  MCTS(string s0, int max_simulations=800,
       float c=1.1414, float w_r=0.5, float w_v=0.75, float w_a=-1.0f);

  Move search();
  bool time_available();
  void simulate(Position position);
  vector<tuple<string, Move> > sim_tree(Position& position);
  Move default_policy(const Position& position);
  float sim_default(Position position);
  void new_node(string s);
  float uct_value(string s, Move a, float c, float w_a, float w_v) const;
  float lookup_Q(string s, Move a) const;
  vector<Move> pv(string s0) const;
  Move select_move(const Position& position, float c) const;
  void backup(const vector<tuple<string, Move> >& states_actions, float z);
  vector<Move> get_actions(string s) const;
  vector<Move> get_actions(const Position& position) const;
  string get_state(const Position& position) const;
  string edge_key(string s, Move a) const;
};

#endif
