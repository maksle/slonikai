#ifndef BB_GAURD
#define BB_GAURD

#include <string>
#include "types.h"

namespace Bitboards {
  void init();
  const std::string print_bb(Bitboard b);
}

extern Bitboard RankBB[RANK_NB];
extern Bitboard FileBB[FILE_NB];
extern Bitboard SquareBB[SQUARE_NB + 1];
extern Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
extern Bitboard LineBB[SQUARE_NB][SQUARE_NB];
extern Bitboard AheadBB[SIDE_NB][SQUARE_NB];

extern Bitboard castlingPathBB[CASTLING_RIGHT_NB];
extern Bitboard castlingCheckPathBB[CASTLING_RIGHT_NB];

extern Bitboard StepAttacksBB[PIECETYPE_NB][SQUARE_NB];
extern Bitboard PseudoAttacksBB[PIECETYPE_NB][SQUARE_NB];

extern const Bitboard WHITE_SQS;
extern const Bitboard DARK_SQS;

extern const Bitboard EMPTY_BOARD;
extern const Bitboard FULL_BOARD;

extern const Bitboard A_FILE;
extern const Bitboard B_FILE;
extern const Bitboard G_FILE;
extern const Bitboard H_FILE;

inline Square lsb(Bitboard b) {
  assert(b);
  return Square(__builtin_ctzll(b));
}

inline Square msb(Bitboard b) {
  assert(b);
  return Square(63 - __builtin_clzll(b));
}

inline int popcount(Bitboard b) {
  return __builtin_popcountll(b);
}

inline Square pop_lsb(Bitboard& b) {
  const Square s = lsb(b);
  b &= b - 1;
  return s;
}

inline Bitboard ls1b(Bitboard p) {
  // least significant 1 bit
  return p & -p;
}

inline Bitboard reset_ls1b(Bitboard p) {
  // flip least significant 1 bit
  return p & (p-1);
}

inline Bitboard step_attacks(PieceType pt, Square sq) {
  return StepAttacksBB[pt][sq];
}

inline Bitboard pseudo_attacks(PieceType pt, Square sq) {
  return PseudoAttacksBB[pt][sq];
}

inline Bitboard between_sqs(Square sq1, Square sq2) {
  return BetweenBB[sq1][sq2];
}

inline Bitboard line_sqs(Square sq1, Square sq2) {
  return LineBB[sq1][sq2];
}

inline Bitboard rank(Rank r) { 
  return RankBB[r];
}

inline Bitboard operator|(Bitboard b, const Square& s) { return b | SquareBB[s]; }
inline Bitboard operator|(Square s, Bitboard b) { return b | SquareBB[s]; }
inline Bitboard operator|(Square s1, Square s2) { return SquareBB[s1] | SquareBB[s2]; }
inline Bitboard& operator|=(Bitboard& b, const Square &s) { return b |= SquareBB[s]; }

inline Bitboard operator&(Bitboard b, const Square& s) { return b & SquareBB[s]; }
inline Bitboard operator&(Square s, Bitboard b) { return b & SquareBB[s]; }
inline Bitboard operator&(Square s1, Square s2) { return SquareBB[s1] & SquareBB[s2]; }
inline Bitboard& operator&=(Bitboard& b, const Square &s) { return b &= SquareBB[s]; }

inline Bitboard operator^(Bitboard b, Square s) { return b ^ SquareBB[s]; }
inline Bitboard& operator^=(Bitboard& b, Square s) { return b ^= SquareBB[s]; }

Bitboard shift_north(Bitboard b, Side side);
Bitboard shift_south(Bitboard b, Side side);

#endif
