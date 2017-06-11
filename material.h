#ifndef MATERIAL_GAURD
#define MATERIAL_GAURD

#include "types.h"

int PhaseWeight[PIECETYPE_NB] = { 0, 0, 1, 1, 2, 4 };
int MaxCount[PIECETYPE_NB] = { 0, 8, 2, 2, 2, 1 };
const int MAX_PHASE = PhaseWeight[PAWN] * 16
  + PhaseWeight[KNIGHT] * 4
  + PhaseWeight[BISHOP] * 4
  + PhaseWeight[ROOK] * 4
  + PhaseWeight[QUEEN] * 2;

Value material_bootstrap(const int counts[PIECE_NB]);
int game_phase(const int counts[PIECE_NB]);
Value phase(const int counts[PIECE_NB]);
int scale_phase(int mg, int eg, int phase);
  
#endif
