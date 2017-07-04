#ifndef MOVEGEN_GAURD_H
#define MOVEGEN_GAURD_H

#include "types.h"
#include <vector>

class Position;

enum GenType {
    ALL_PSEUDO,
    ALL_LEGAL,
    EVASIONS,
    QUIESCENCE_TIER1,
    QUIESCENCE_TIER2
};

struct ProbMove {
  Move move;
  double probability;

  ProbMove(Move m, double p) : move(m), probability(p) {};
  
  void operator=(Move m) { move = m; };
  operator Move() const { return move; };

  bool operator<(const ProbMove& m) const {
    return probability < m.probability;
  }
};

Bitboard knight_attack(Square sq);
Bitboard king_attack(Square sq);
Bitboard pawn_attack(Square sq, Side side);
Bitboard bishop_attack(Square sq, Bitboard occupied = 0);
Bitboard rook_attack(Square sq, Bitboard occupied = 0);
Bitboard queen_attack(Square sq, Bitboard occupied = 0);
Bitboard piece_attack(Piece pt, Square sq, Bitboard occupied = 0);

template <GenType GT>
std::vector<Move>& generate(const Position&, std::vector<Move>&);

template<CastlingRight cr>
std::vector<Move>& generate_castling(const Position&, std::vector<Move>&);

std::vector<ProbMove> evaluate_moves(const Position&, const std::vector<Move>&);

template<GenType GT>
struct MoveGen {
  std::vector<Move> moves;
  const Position* position = nullptr;
  
  MoveGen(const Position& pos) {
    position = &pos;
    generate<GT>(pos, moves);
  }
  std::vector<ProbMove> evaluate() { return evaluate_moves(*position, moves); }
};

#endif
