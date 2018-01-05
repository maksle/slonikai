#ifndef MCTS_H_GUARD
#define MCTS_H_GUARD

#include <vector>
#include <map>
#include <string>
#include <functional>
#include <cmath> // sqrt, log
#include <random>
#include "types.h"

using namespace std;

class Position;

const int NO_EVAL = 10;

struct MCTSEdge {
  int N = 0;
  int Q = 0;
  float Pnn = 0;
};

struct MCTSNode {
  int N = 0;
  float Vnn = 0;
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
  map<string, MCTSNode> tree;
  int simulations;
  int max_simulations;
 public:

  MCTS(string s0, int max_simulations=800,
       float c=1.1414, float w_r=0.5, float w_v=0.75, float w_a=-1.0f);

  Move search();
  bool time_available();
  void simulate(Position position);
  vector<tuple<string, Move> > sim_tree(Position position);
  Move default_policy(Position position);
  float sim_default(Position position);
  void new_node(string s);
  float uct_value(string s, Move a, float c, float w_a, float w_v);
  float lookup_Q(string s, Move a);
  vector<Move> pv(string s0);
  Move select_move(Position position, float c);
  void backup(vector<tuple<string, Move> > states_actions, float z);
  vector<Move> get_actions(string s);
  vector<Move> get_actions(Position position);
  string get_state(const Position& position);
};

template<typename T>
typename vector<T>::const_iterator
random_choice(const vector<T>& choices) {
  std::random_device rd;
  std::mt19937 mt(rd());
  /* std::cout << ":" << choices.size() << std::endl; */
  std::uniform_int_distribution<> dis(0, choices.size() - 1);
  auto it = choices.begin();
  std::advance(it, dis(mt));
  return it;
}

#endif
