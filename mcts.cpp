#include <iostream>
#include "mcts.h"
#include "position.h"
#include "bb.h"


enum GameTerminationReason {
   NONE, NO_MOVES, INSUFFICIENT_MATERIAL, NO_PROGRESS
};

string str_moves(vector<Move> moves) {
   string str = "";
   for (const auto& move : moves) {
        Square from = from_sq(move);
        Square to = to_sq(move);
		str += char(file_of(from) + 'a') + char(rank_of(from) + '1') + char(file_of(to) + 'a') + char(rank_of(to) + '1');
        str += " ";
    }
   return str;
}

std::ostream& operator<<(std::ostream& os, std::vector<Move> moves) {
    for (const auto& move : moves) {
        Square from = from_sq(move);
        Square to = to_sq(move);
        os << move << " ";
    }
    os << "\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, vector<MCTSNode*> path) {
   for (const auto& it: path) {
	  os << it->move << " (" << it << ") ";
   }
   os << "leaf Q: (" << path.back()->Q << ") " << endl;
   return os;
}

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

   root_node = MCTSNode();
   root_node.repr = s0;
}

MCTSNode* MCTS::search() {
   while (time_available()) {
	  Position root_position = Position(s0);
	  simulate(root_node, root_position);
   }
   Position root_position = Position(s0);
   return recover_move(&root_node, root_position);
}

bool MCTS::time_available() {
   return simulations < max_simulations;
}

void MCTS::simulate(MCTSNode& node, Position& position) {
   simulations++;
   vector<MCTSNode*> path = sim_tree(&node, position);

   float z;
   if (w_r > 0) {
	  Position position_copy = Position(position.fen());
	  z = sim_default(position_copy);
	  // m_z = ()m_z + z)
   }
   else
	  z = 0;
   backup(path, z);
}

vector<MCTSNode*> MCTS::sim_tree(MCTSNode* node, Position& position) {
   vector<MCTSNode*> path { node };

   MCTSNode* curr_node = node;
   GameTerminationReason reason;
   while (!game_over(position, reason))
   {
      if (!curr_node->expanded) {
         expand_node(curr_node, position);
         break;
      }
      curr_node = select_move(curr_node, position, this->c);
      path.push_back(curr_node);
      position.make_move(curr_node->move);
   }
   return path;
}


Move MCTS::default_policy(const Position& position) {
   vector<Move> legal = MoveGen<ALL_LEGAL>(position).moves;
   return *(random_choice.select<>(legal));
}

float MCTS::sim_default(Position& position) {
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

void MCTS::expand_node(MCTSNode* node, const Position& position)
{
   // cout << "Expand Node" << "\n";
   // cout << "expanding " << node << " " << node->move << "\n";
   // cout << position;
   bool usingNN = this->w_a || this->w_v;

   PositionEvaluation evals;
   float v;
   vector<float> probs;
   if (usingNN) {
	  evals = evaluator(position);
	  v = std::get<0>(evals);
	  probs = std::get<1>(evals);
   }
   
   vector<Move> legal = get_actions(position);
   for (const auto& m : legal)
   {
	  MCTSNode* edge = new MCTSNode();
	  edge->move = m;
	  edge->Q = 0;
	  edge->N = 0;

	  edge->Pnn = 0;
	  // if (usingNN && this->w_a) {
	  // 	 int index = 0;
	  // 	 edge->Pnn = probs[index];
	  // }

	  // assert(position.is_legal(edge->move));
	  // cout << "Adding child " << edge << " " << edge->move << endl;
	  node->add_child(edge);
   }

   // cout << "Checking what we've added" << endl;
   // MCTSNode* child = node->first_child;
   // MCTSNode::Iterator iter(child);
   // for (; !iter.end(); child = iter.next())
   // {
   // 	  Move a = child->move;
   // 	  cout << a << " ";
   // }
   // cout << "\n";

   node->expanded = true;
}

// string MCTS::edge_key(string s, Move a) const {
//     Square from = from_sq(a);
//     Square to = to_sq(a);
//     MoveType mt = type_of(a);

//     string res = s;
//     res += char(file_of(from) + 'a');
//     res += char(rank_of(from) + '1');
//     res += char(file_of(to) + 'a');
//     res += char(rank_of(to) + '1');
//     if (mt == PROMOTION) {
//         PieceType bt = promo_piece(a);
//         if (bt != QUEEN && bt != PIECETYPE_NONE) {
//             res += PieceToChar[bt + 6 /*lowercase*/];
//         }
//     }
//     return res;
// }

// float MCTS::lookup_Q(string s, Move a) const {
//     return (tree.find(edge_key(s, a))->second).Q;
// }

vector<Move> MCTS::pv() {
   vector<Move> actions;

   MCTSNode* node = &root_node;
   
   Position position = Position(node->repr);
   
   MCTSNode* curr_node = node;
   while (curr_node->expanded) {
	  curr_node = recover_move(curr_node, position);
	  if (!curr_node)
		 break;
	  Move a = curr_node->move;
	  // cout << position.fen();
	  position.make_move(a);
	  // cout << position;
	  actions.push_back(a);
   }
   return actions;
}

MCTSNode* MCTS::recover_move(MCTSNode* node, const Position& position) const {
   Side stm = position.side_to_move();
   float best_val = NO_EVAL;
   MCTSNode* best_node;
   
   MCTSNode* child = node->first_child;
   MCTSNode::Iterator iter(child);
   for (; !iter.end(); child = iter.next())
   {
	  Move a = child->move;
	  float score = child->Q;

	  if (stm == BLACK)
		 score = -score;
	  
	  if (best_val == NO_EVAL || score > best_val)
	  {
		 best_val = score;
		 best_node = child;
	  }
   }
   return best_node;
}

MCTSNode* MCTS::select_move(MCTSNode* node, const Position& position, float c) const {
   Side stm = position.side_to_move();
   float best_val = NO_EVAL;
   MCTSNode* best_node;

   // string smoves = str_moves(position.moves);
   // if (smoves == "d5d3 h7h8 f7f8 ") {
   // 	  cout << position;
   // 	  cout << position.fen() << "\n";
   // 	  cout << position.moves.size() << "\n";
   // 	  cout << position.moves << "\n";
   // }
    
   // cout << "Select Move" << "\n";
   // cout << "expanded: " << node->expanded << "\n";
   // cout << position;
   // cout << position.fen() << "\n";
   // cout << position.moves.size() << "\n";
   // cout << position.moves;
   // cout << "Legal: " << get_actions(position); 
   // cout << "Options: ";
   
   MCTSNode* child = node->first_child;
   MCTSNode::Iterator iter(child);
   for (; !iter.end(); child = iter.next())
   {
	  Move a = child->move;
	  // cout << a << " ";

	  float uct = 1000;
	  if (child->N > 0) {
		 uct = this->c * sqrt(log(node->N) / child->N);
	  }
   
	  float policy_prior_bonus = 0;
	  if (this->w_a)
		 policy_prior_bonus = w_a * child->Pnn / (child->N + 1);

	  float value_prior_bonus = 0;
	  // if (w_v)
	  //     value_prior_bonus = w_v * s_prime_node.Vnn;

	  float exploration = uct + policy_prior_bonus;
	  float value = child->Q + value_prior_bonus;
	  
	  // cout << child << " Q " << child->Q << ", UCT " << uct << endl;
	  
	  if (stm == BLACK)
		 value = -value;

	  float score = value + exploration;

	  // cout << value << " " << exploration << "\n";
	  
	  if (best_val == NO_EVAL || score > best_val)
	  {
		 best_val = score;
		 best_node = child;
	  }
   }

   // assert(position.is_legal(best_node->move));
   
   // cout << "\nChose node " << best_node << " " << best_node->move << "\n";
   return best_node;
}

void MCTS::backup(vector<MCTSNode*>& path, float z)
{
   // cout << z;
   for (vector<MCTSNode*>::iterator it = path.begin(); it != path.end(); ++it)
   {
	  MCTSNode* node = *it;
	  node->N++;
	  float v = z;
	  node->Q += (v - node->Q) / node->N;
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
