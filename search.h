#ifndef SEARCH_GAURD_H
#define SEARCH_GAURD_H

#include <vector>
#include <limits>
#include "types.h"
#include "tt.h"

const int MAX_PLY = 64;

const Score DRAW_VALUE = 0;

const Score NEGATIVE_INF = std::numeric_limits<Score>::lowest();
const Score POSITIVE_INF = std::numeric_limits<Score>::max();

const Score MATE_LOSE = NEGATIVE_INF + 1000;
const Score MATE_WIN = POSITIVE_INF - 1000;

class Position;

struct RootMove {
  Move move = MOVE_NONE;
  Score value = NEGATIVE_INF;
};

struct Signals {
  bool stop = false;
};

struct Limits {
  int max_depth = 64;
};

struct TrainUpdateNode {
  std::string fen;
  Score leaf_eval;
  TTBound bound;
  Score static_eval;
};

struct TrainMeta {
  bool training = false;
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
    uint64_t total_allowance;
  };

  struct SearchOutput
  {
    Score value;
    std::vector<Move> pv;
  };
  
  const uint64_t MIN_PVS_ALLOWANCE = 16;

  template<bool Root = true> int perft(Position& pos, int depth);

  /* template<bool PVNode> */
  /*   int search(Position& pos, SearchInfo* si, int alpha, int beta, uint64_t allowance); */

  /* template<bool PVNode> */
  /*   int qsearch(Position& pos, SearchInfo* si, int alpha, int beta); */
  
  SearchOutput iterative_deepening(Context& context);
  
}

#endif
