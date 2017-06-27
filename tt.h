#ifndef TT_H_GAURD
#define TT_H_GAURD

#include <stdint.h>
#include "types.h"

enum TTBound {
  LOW_BOUND,
  HIGH_BOUND,
  EXACT_BOUND
};

class TTable;

class TTEntry {
  friend TTable;
 public:
  Move best_move() const { return static_cast<Move>(move); }
  Score score() const { return static_cast<Score>(value); }
  TTBound bound() const { return TTBound(gen_bound & 0x3); }
  uint16_t gen() const { return gen_bound >> 2; }
  
  uint64_t allowance;
  uint64_t key;
  uint16_t move;
  uint16_t value;
  uint16_t gen_bound;
};

class TTable {

 public:
  TTable() : current_generation(4) {};

  TTable(const TTable&) = delete;
  TTable& operator=(const TTable&) = delete;

  void resize(size_t size);
  TTEntry* probe(Key zkey);
  void prefetch(Key zkey);
  void save(const Position& pos, Move move, TTBound bound, Score val, uint64_t allowance);
  void age();
  void clear();

 private:
  std::vector<TTEntry> entries;
  uint16_t current_generation;
};

extern TTable TT;

#endif
