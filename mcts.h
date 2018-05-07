#ifndef MCTS_H_GUARD
#define MCTS_H_GUARD

#include <vector>
#include <unordered_set>
#include <string>
#include <functional>
#include <cmath> // sqrt, log
#include <random>
#include "types.h"
#include "movegen.h"

using namespace std;

class Position;

const int NO_EVAL = 10000;

struct RaveOptions {
  bool white_rave = false;
  bool black_rave = false;
};

class MCTSNode {
 public:
  /* vector<MCTSNode*> children; */
  MCTSNode* parent = nullptr;
  MCTSNode* first_child = nullptr;
  MCTSNode* sibling = nullptr;
  MCTSNode* prev_sibling = nullptr;

  bool expanded = false;
  Move move; 
  string repr;

  int N = 0;
  int Q = 0;

  int N_RAVE = 0;
  int Q_RAVE = 0;

  float Vnn = 0;
  float Pnn = 0;

  void add_child(MCTSNode* node) {
    node->sibling = this->first_child;
    if (this->first_child)
      this->first_child->prev_sibling = node;
    this->first_child = node;
  }

  ~MCTSNode() {
    if (parent && parent->first_child == this)
      parent->first_child = sibling;
    if (prev_sibling && sibling)
        prev_sibling->sibling = sibling;
    if (first_child)
      delete first_child;
  }
  
  class Iterator
  {
  public:
  Iterator(MCTSNode* begin) :
    iter(begin) {}
    MCTSNode* next() {
      iter = iter->sibling;
      return iter;
    }
    bool end() {
      if (iter == nullptr)
        return true;
      /* if (iter->sibling == nullptr) */
      /*   return true; */
      return false;
      /* return (!iter || !iter->sibling); */
    }
  private:
    MCTSNode* iter;
  };
  
};

std::ostream& operator<<(std::ostream& os, vector<MCTSNode*> path);

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

typedef tuple<float, vector<float>> PositionEvaluation; // value, policy
typedef std::function<PositionEvaluation(const Position&)> MCTSEvaluator;

class MCTS {
 private:
  /* string s0; */
  MCTSNode root_node;
  float c;
  float w_r;
  float w_a;
  float w_v;
  MCTSEvaluator evaluator;
  /* unordered_map<string, MCTSNode> tree; */
  int simulations;
  int max_simulations;
  RandomSelector random_choice;

  RaveOptions rave_options;

 public:

  MCTS(string s0,int max_simulations=800,
       float c=1.1414, float w_r=0.5, float w_v=0.75, float w_a=-1.0f,
       RaveOptions rave_options=RaveOptions());
  
  MCTSNode* search(MCTSEvaluator&);
  bool time_available();
  void simulate(MCTSNode&, Position&);
  vector<MCTSNode*> sim_tree(MCTSNode* node, Position& position);
  Move default_policy(const Position& position);
  float sim_default(Position& position);
  /* void new_node(string s); */
  void expand_node(MCTSNode* parent, const Position& position);
  /* float lookup_Q(string s, Move a) const; */
  vector<Move> pv();
  MCTSNode* select_move(MCTSNode* node, const Position& position, float c);
  MCTSNode* recover_move(MCTSNode* node, const Position& position) const;
  void backup(vector<MCTSNode*>& path, float z);
  vector<Move> get_actions(string s) const;
  vector<Move> get_actions(const Position& position) const;
  string get_state(const Position& position) const;
  /* string edge_key(string s, Move a) const; */
  void play(Move move, Position& pos);
};

int playMCTS(string fen, int sims, bool white_rave);

#endif
