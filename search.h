#ifndef SEARCH_GAURD_H
#define SEARCH_GAURD_H

#include <vector>
#include "types.h"
#include "tt.h"

class Position;

struct RootMove {
  Move move;
  Score value;
};

struct Signals {
  bool stop;
};

struct Limits {
  int max_depth;
};

struct TrainUpdateNode {
  std::string fen;
  Score leaf_eval;
  TTBound bound;
  Score static_eval;
};

struct TrainMeta {
  bool training;
  std::vector<TrainUpdateNode> search_states;
};

namespace Search { 

  struct Context
  {
    Position root_position;
    std::vector<RootMove> root_moves;
    Signals signals;
    Limits limits;
    TrainMeta train;
  };
  
  struct SearchInfo
  {
    int ply;
    std::vector<Move> pv;
    int static_eval;
    bool is_QS;
    double total_allowance;
  };

  struct SearchOutput
  {
    Score value;
    std::vector<Move> pv;
  };
  
  const double MIN_PVS_ALLOWANCE = 16;

  template<bool Root = true> int perft(Position& pos, int depth);

  /* template<bool PVNode> */
  /*   int search(Position& pos, SearchInfo* si, int alpha, int beta, double allowance); */

  /* template<bool PVNode> */
  /*   int qsearch(Position& pos, SearchInfo* si, int alpha, int beta); */
  
  SearchOutput iterative_deepening(Context& context);
  
}

#endif
